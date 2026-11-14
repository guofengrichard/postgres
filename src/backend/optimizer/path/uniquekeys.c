/*-------------------------------------------------------------------------
 *
 * uniquekeys.c
 *	  Utilities for proving relations distinct using UniqueKeys
 *
 * A UniqueKey is a set of EquivalenceClasses that no two rows of a base
 * relation agree on.  Join relations have no UniqueKeys of their own:
 * whether a join relation is distinct is worked out when needed, from the
 * UniqueKeys of the base relations it consists of.
 *
 * There are two kinds of question.  Join clauses are strict, so when matching
 * them we need not care about rows that have NULLs in the columns of
 * interest.  DISTINCT and GROUP BY, on the other hand, consider NULLs equal
 * to each other, so for them only UniqueKeys that hold on those terms can be
 * used.
 *
 * UniqueKeys always describe the rows that a base relation really has, and
 * the ECs in them are those of the relation's columns as they are when it is
 * scanned.  A join relation that includes outer joins can also emit
 * null-extended rows for its base relations; joinrel_is_distinct_for_ecs()
 * explains how that is dealt with.
 *
 * See src/backend/optimizer/README for more information.
 *
 *
 * Portions Copyright (c) 1996-2026, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * IDENTIFICATION
 *	  src/backend/optimizer/path/uniquekeys.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/sysattr.h"
#include "nodes/multibitmapset.h"
#include "optimizer/clauses.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "parser/parsetree.h"
#include "rewrite/rewriteManip.h"
#include "utils/lsyscache.h"

static bool rel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *rel,
									Bitmapset *known_ecs, bool nulls_equal);
static bool joinrel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *joinrel,
										Bitmapset *known_ecs,
										bool nulls_equal);
static bool get_semijoin_rhs_relids(PlannerInfo *root, RelOptInfo *joinrel,
									Relids *rhs_relids);
static Bitmapset *get_left_join_ecs(PlannerInfo *root, RelOptInfo *joinrel);
static Bitmapset *get_joined_ecs(PlannerInfo *root, RelOptInfo *joinrel,
								 Bitmapset *left_join_ecs);
static void build_base_rel_uniquekeys(PlannerInfo *root);
static void build_base_eclasses(PlannerInfo *root);
static void build_plain_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel);
static void build_subquery_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel);
static bool add_subquery_uniquekey(RelOptInfo *rel, Query *subquery,
								   List *distinct_cols, List *col_ecs,
								   List *const_cols, Bitmapset *colnos);
static void build_subquery_scanjoin_uniquekeys(PlannerInfo *root,
											   RelOptInfo *rel,
											   List *distinct_cols,
											   List *col_ecs,
											   List *const_cols);
static int	find_output_ec_index(PlannerInfo *root, DistinctColInfo *dcinfo);
static void build_notnull_eclass_indexes(PlannerInfo *root, RelOptInfo *rel);
static void add_uniquekey(RelOptInfo *rel, Bitmapset *eclass_indexes,
						  bool null_aware);
static int	find_ec_index(PlannerInfo *root, RelOptInfo *rel, Expr *expr,
						  List *opfamilies, Oid collation);
static Bitmapset *add_base_ec_index(PlannerInfo *root,
									Bitmapset *eclass_indexes,
									EquivalenceClass *ec);
static Expr *get_base_expr(PlannerInfo *root, Expr *expr, RelOptInfo **rel);
static Expr *get_output_base_expr(PlannerInfo *root, TargetEntry *tle,
								  RelOptInfo **rel);
static bool rel_has_uniquekey_in(PlannerInfo *root, RelOptInfo *rel,
								 Bitmapset *eclass_indexes,
								 Bitmapset *joined_ecs,
								 bool nulls_equal);
static bool full_nullable_rel_is_marked(PlannerInfo *root, RelOptInfo *rel,
										Bitmapset *eclass_indexes,
										bool nulls_equal);
static bool rel_has_notnull_column_in(PlannerInfo *root, RelOptInfo *rel,
									  Bitmapset *eclass_indexes);
static Bitmapset *get_own_eclass_indexes(PlannerInfo *root, RelOptInfo *rel);


/*
 * joinrel_supports_distinctness
 *		Could the join relation possibly be proven distinct?
 *
 * This is a pre-checking function for joinrel_is_distinct_for(), in the same
 * way as rel_supports_distinctness() is for rel_is_distinct_for().
 */
bool
joinrel_supports_distinctness(PlannerInfo *root, RelOptInfo *joinrel)
{
	Relids		rhs_relids;

	Assert(IS_JOIN_REL(joinrel));

	/* For a child join, we look at its parent; see joinrel_is_distinct_for */
	if (IS_OTHER_REL(joinrel))
		joinrel = joinrel->top_parent;

	/* A grouped join rel's rows are not combinations of base rel rows */
	if (IS_GROUPED_REL(joinrel))
		return false;

	build_base_rel_uniquekeys(root);

	/* If every base rel in the join has a UniqueKey, we can go ahead */
	if (!bms_overlap(joinrel->relids, root->no_uniquekey_relids))
		return true;

	/* Otherwise, those that don't must not need one */
	if (!get_semijoin_rhs_relids(root, joinrel, &rhs_relids))
		return false;

	return bms_is_subset(bms_intersect(joinrel->relids,
									   root->no_uniquekey_relids),
						 rhs_relids);
}

/*
 * joinrel_is_distinct_for
 *		Does the join relation return only distinct rows according to
 *		clause_list?
 *
 * clause_list is as for rel_is_distinct_for(): each clause is a mergejoinable
 * equality with an expression of this relation on one side, as identified by
 * its outer_is_left flag, and an expression not involving this relation on
 * the other.  Return true if no two rows emitted by the join relation could
 * possibly join to the same row of the other relation.
 */
bool
joinrel_is_distinct_for(PlannerInfo *root, RelOptInfo *joinrel,
						List *clause_list)
{
	Bitmapset  *known_ecs = NULL;

	if (!joinrel_supports_distinctness(root, joinrel))
		return false;

	/*
	 * A child join's rows are a subset of its parent's rows, and its clauses
	 * use the same ECs as its parent's.  So it is distinct if its parent is,
	 * and we need no UniqueKeys for child relations.
	 */
	if (IS_OTHER_REL(joinrel))
		joinrel = joinrel->top_parent;

	foreach_node(RestrictInfo, rinfo, clause_list)
	{
		/* caller identified the side belonging to this relation for us */
		if (rinfo->outer_is_left)
			known_ecs = add_base_ec_index(root, known_ecs, rinfo->right_ec);
		else
			known_ecs = add_base_ec_index(root, known_ecs, rinfo->left_ec);
	}

	/* The clauses are strict, so rows with NULLs are of no concern */
	return joinrel_is_distinct_for_ecs(root, joinrel, known_ecs, false);
}

/*
 * subquery_rel_supports_distinctness
 *		Could the subquery relation possibly be proven distinct by its
 *		UniqueKeys?
 *
 * This is a pre-checking function for subquery_rel_is_distinct_for().
 */
bool
subquery_rel_supports_distinctness(PlannerInfo *root, RelOptInfo *rel)
{
	Assert(rel->rtekind == RTE_SUBQUERY);

	/*
	 * The UniqueKeys of interest follow from the subquery's own relations, so
	 * the subquery must have been planned.  That also keeps us from building
	 * UniqueKeys before join removal is done.  Appendrel children have none.
	 */
	if (rel->reloptkind != RELOPT_BASEREL || rel->subroot == NULL)
		return false;

	build_base_rel_uniquekeys(root);

	return rel->uniquekeys != NIL;
}

/*
 * subquery_rel_is_distinct_for
 *		Does the subquery relation return only distinct rows according to
 *		clause_list?
 *
 * clause_list is as for joinrel_is_distinct_for().
 *
 * query_is_distinct_for() answers this for a subquery that is distinct by
 * construction.  Here we use the relation's UniqueKeys, which also cover a
 * subquery that is distinct because of the relations it is made of.
 *
 * If the subquery has LATERAL references, its UniqueKeys hold only within one
 * scan.  That is good enough, as query_is_distinct_for() also assumes: the
 * rows that could join to a given row of the other relation all come from the
 * same scan.
 */
bool
subquery_rel_is_distinct_for(PlannerInfo *root, RelOptInfo *rel,
							 List *clause_list)
{
	Bitmapset  *known_ecs = NULL;

	if (!subquery_rel_supports_distinctness(root, rel))
		return false;

	foreach_node(RestrictInfo, rinfo, clause_list)
	{
		/* caller identified the side belonging to this relation for us */
		if (rinfo->outer_is_left)
			known_ecs = add_base_ec_index(root, known_ecs, rinfo->right_ec);
		else
			known_ecs = add_base_ec_index(root, known_ecs, rinfo->left_ec);
	}

	/* The clauses are strict, so rows with NULLs are of no concern */
	return rel_has_uniquekey_in(root, rel, known_ecs, NULL, false);
}

/*
 * rel_is_distinct_for_pathkeys
 *		Does the relation return only distinct rows according to the given
 *		pathkeys?
 *
 * Return true if no two rows emitted by the relation could possibly have
 * equal values in all the pathkeys' ECs.  If nulls_equal is true, NULLs are
 * considered equal to each other; otherwise rows having a NULL in any of the
 * ECs are ignored.  The sort ordering of the pathkeys is of no interest.
 *
 * An empty list means all rows are considered equal, so the relation is then
 * distinct only if it has at most one row.
 */
bool
rel_is_distinct_for_pathkeys(PlannerInfo *root, RelOptInfo *rel,
							 List *pathkeys, bool nulls_equal)
{
	Bitmapset  *known_ecs = NULL;

	/* As in joinrel_is_distinct_for, look at the parent of a child rel */
	if (IS_OTHER_REL(rel))
		rel = rel->top_parent;

	/* We know nothing about relations other than these */
	if (!IS_JOIN_REL(rel) && rel->reloptkind != RELOPT_BASEREL)
		return false;

	/* add_base_ec_index() needs the ECs that this creates */
	build_base_rel_uniquekeys(root);

	foreach_node(PathKey, pathkey, pathkeys)
	{
		known_ecs = add_base_ec_index(root, known_ecs, pathkey->pk_eclass);
	}

	return rel_is_distinct_for_ecs(root, rel, known_ecs, nulls_equal);
}

/*
 * rel_is_distinct_for_ecs
 *		Does the base or join relation return only distinct rows according to
 *		the given set of ECs?
 *
 * This is as joinrel_is_distinct_for_ecs(), but accepts a base relation too.
 */
static bool
rel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *rel,
						Bitmapset *known_ecs, bool nulls_equal)
{
	if (IS_JOIN_REL(rel))
		return joinrel_supports_distinctness(root, rel) &&
			joinrel_is_distinct_for_ecs(root, rel, known_ecs, nulls_equal);

	if (rel->reloptkind != RELOPT_BASEREL)
		return false;

	build_base_rel_uniquekeys(root);

	return rel_has_uniquekey_in(root, rel, known_ecs, NULL, nulls_equal);
}

/*
 * joinrel_is_distinct_for_ecs
 *		Does the join relation return only distinct rows according to the
 *		given set of ECs?
 *
 * Return true if no two rows emitted by the join relation could possibly
 * have equal values in all the ECs of known_ecs, which is a set of indexes
 * into root->eq_classes.  If nulls_equal is true, NULLs are considered equal
 * to each other; otherwise rows having a NULL in any of the ECs are ignored.
 *
 * A join emits each combination of its base relations' rows at most once.
 * So the join is distinct if fixing the values of the known ECs leaves only
 * one possible row for each of its base relations.  A base relation with a
 * UniqueKey within the set of known ECs can have only one row, and so it
 * fixes the value of every EC that has a member belonging to it; those ECs
 * are added to the set.  We repeat this until all base relations are
 * accounted for or we can make no more progress.
 *
 * If outer joins are performed within the join relation, a base relation can
 * also be represented in a join row by a null-extended row.  That is just
 * one more possibility for it, and the above holds as it stands if we take
 * "row" to include that, provided something decides between a relation's
 * real row and its null-extended row (see rel_has_uniquekey_in).  The known
 * ECs are always those of un-nulled expressions, which is what UniqueKeys
 * consist of; the caller must have translated the ECs of nulled expressions
 * (see add_base_ec_index).
 *
 * A left join pairs each row of its left-hand side either with the
 * right-hand rows that satisfy the join clauses, or else with a
 * null-extended row, but never with both.  Hence, once the left-hand row is
 * fixed, so are the values of the right-hand ECs that the clauses equate to
 * the left-hand side, for as far as there are matching rows at all.  We
 * treat those ECs as known from the start.  That is safe because we succeed
 * only if the left-hand relations get fixed too, and nothing on the
 * right-hand side can help with that: no EC has un-nulled members on both
 * sides of a left join.  get_left_join_ecs collects those ECs.
 *
 * A semijoin or antijoin emits each row of its left-hand side at most once,
 * whatever its right-hand side holds.  So the relations on the right-hand
 * side of such a join performed within the join relation need not be
 * accounted for.  They are found by get_semijoin_rhs_relids.
 *
 * The caller must have checked joinrel_supports_distinctness().
 */
static bool
joinrel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *joinrel,
							Bitmapset *known_ecs, bool nulls_equal)
{
	Bitmapset  *left_join_ecs;
	Bitmapset  *joined_ecs = NULL;
	Relids		full_join_relids = NULL;
	Relids		rhs_relids;
	Relids		remaining;
	bool		found;

	if (!get_semijoin_rhs_relids(root, joinrel, &rhs_relids))
		return false;

	left_join_ecs = get_left_join_ecs(root, joinrel);
	known_ecs = bms_add_members(known_ecs, left_join_ecs);

	/*
	 * When NULLs are considered equal, a UniqueKey that is not null_aware can
	 * still be used if join clauses keep its ECs from being NULL; see
	 * rel_has_uniquekey_in().  So find those ECs.
	 */
	if (nulls_equal)
		joined_ecs = get_joined_ecs(root, joinrel, left_join_ecs);

	/*
	 * Relations that are nulled by a full join need extra care; see
	 * full_nullable_rel_is_marked().  So find the full joins performed within
	 * the join relation.
	 */
	if (bms_overlap(joinrel->relids, root->outer_join_rels))
	{
		foreach_node(SpecialJoinInfo, sjinfo, root->join_info_list)
		{
			if (sjinfo->jointype == JOIN_FULL &&
				bms_is_member(sjinfo->ojrelid, joinrel->relids))
				full_join_relids = bms_add_member(full_join_relids,
												  sjinfo->ojrelid);
		}
	}

	/*
	 * The relids of outer joins are not relations to account for, and neither
	 * are the right-hand relations of semijoins and antijoins.
	 */
	remaining = bms_difference(joinrel->relids, root->outer_join_rels);
	remaining = bms_del_members(remaining, rhs_relids);

	do
	{
		int			relid = -1;

		found = false;

		while ((relid = bms_next_member(remaining, relid)) >= 0)
		{
			RelOptInfo *rel = find_base_rel(root, relid);

			/* A rel nulled by a full join within this join rel needs more */
			if (bms_overlap(rel->nulling_relids, full_join_relids) &&
				!full_nullable_rel_is_marked(root, rel, known_ecs,
											 nulls_equal))
				continue;

			/*
			 * A relation with lateral references is scanned anew for each row
			 * of the relations it refers to, and its UniqueKeys hold only
			 * within one scan.  So those relations have to come first.
			 */
			if (bms_overlap(rel->lateral_relids, remaining))
				continue;

			if (!rel_has_uniquekey_in(root, rel, known_ecs, joined_ecs,
									  nulls_equal))
				continue;

			remaining = bms_del_member(remaining, relid);
			known_ecs = bms_add_members(known_ecs,
										get_own_eclass_indexes(root, rel));
			found = true;
		}
	} while (found && remaining != NULL);

	return remaining == NULL;
}

/*
 * get_semijoin_rhs_relids
 *		Find the relations that are on the right-hand side of a semijoin or
 *		antijoin performed within the join relation.
 *
 * The relids are returned in *rhs_relids.  Returns false if the join
 * relation is one that we cannot reason about: that is the case if it
 * includes the right-hand side of a semijoin but only part of its left-hand
 * side, which means the right-hand side has been unique-ified.
 */
static bool
get_semijoin_rhs_relids(PlannerInfo *root, RelOptInfo *joinrel,
						Relids *rhs_relids)
{
	*rhs_relids = NULL;

	foreach_node(SpecialJoinInfo, sjinfo, root->join_info_list)
	{
		if (sjinfo->jointype != JOIN_SEMI && sjinfo->jointype != JOIN_ANTI)
			continue;

		/*
		 * The join is of no concern if the join relation has nothing of its
		 * right-hand side, or lies entirely within its right-hand side.
		 */
		if (!bms_overlap(sjinfo->syn_righthand, joinrel->relids) ||
			bms_is_subset(joinrel->relids, sjinfo->syn_righthand))
			continue;

		/* Otherwise the whole join must be performed within the relation */
		if (!bms_is_subset(sjinfo->syn_righthand, joinrel->relids) ||
			!bms_is_subset(sjinfo->min_lefthand, joinrel->relids))
			return false;

		*rhs_relids = bms_add_members(*rhs_relids, sjinfo->syn_righthand);
	}

	return true;
}

/*
 * get_left_join_ecs
 *		Get the indexes of the ECs on the nullable side of the mergejoinable
 *		clauses of the left joins performed within the join relation.
 *
 * See joinrel_is_distinct_for_ecs() for what these are good for.
 */
static Bitmapset *
get_left_join_ecs(PlannerInfo *root, RelOptInfo *joinrel)
{
	Bitmapset  *result = NULL;

	if (!bms_overlap(joinrel->relids, root->outer_join_rels))
		return NULL;

	/* In left_join_clauses, the nullable side is on the right */
	foreach_node(OuterJoinClauseInfo, ojcinfo, root->left_join_clauses)
	{
		if (ojcinfo->sjinfo->jointype == JOIN_LEFT &&
			bms_is_member(ojcinfo->sjinfo->ojrelid, joinrel->relids))
			result = add_base_ec_index(root, result,
									   ojcinfo->rinfo->right_ec);
	}

	/* In right_join_clauses, the nullable side is on the left */
	foreach_node(OuterJoinClauseInfo, ojcinfo, root->right_join_clauses)
	{
		if (ojcinfo->sjinfo->jointype == JOIN_LEFT &&
			bms_is_member(ojcinfo->sjinfo->ojrelid, joinrel->relids))
			result = add_base_ec_index(root, result,
									   ojcinfo->rinfo->left_ec);
	}

	return result;
}

/*
 * get_joined_ecs
 *		Get the indexes of the ECs that join clauses performed within the
 *		join relation keep from being NULL.
 *
 * Join clauses are strict, so an expression that a join clause equates to
 * something is not NULL in any row that passes the clause.  That goes for an
 * EC with two members that can be computed within the join relation, since
 * the join enforces their equality, and for the ECs on the nullable side of
 * left-join clauses, which are passed in as left_join_ecs.  (A null-extended
 * row doesn't pass the clause, but such a row is not one of the relation's
 * own rows.)
 *
 * This is only of interest for UniqueKeys that are not null_aware, so to
 * save work we look at just the ECs in those.
 */
static Bitmapset *
get_joined_ecs(PlannerInfo *root, RelOptInfo *joinrel,
			   Bitmapset *left_join_ecs)
{
	Bitmapset  *result = NULL;
	int			relid = -1;

	while ((relid = bms_next_member(joinrel->relids, relid)) >= 0)
	{
		RelOptInfo *rel;

		if (bms_is_member(relid, root->outer_join_rels))
			continue;
		rel = find_base_rel(root, relid);

		foreach_node(UniqueKey, ukey, rel->uniquekeys)
		{
			int			i = -1;

			if (ukey->null_aware)
				continue;

			while ((i = bms_next_member(ukey->eclass_indexes, i)) >= 0)
			{
				EquivalenceClass *ec;
				int			nmembers = 0;

				if (bms_is_member(i, result))
					continue;

				if (bms_is_member(i, left_join_ecs))
				{
					result = bms_add_member(result, i);
					continue;
				}

				/* The members of a broken EC are not known to be equal */
				ec = list_nth_node(EquivalenceClass, root->eq_classes, i);
				if (ec->ec_broken)
					continue;

				foreach_node(EquivalenceMember, em, ec->ec_members)
				{
					if (!em->em_is_const &&
						bms_is_subset(em->em_relids, joinrel->relids) &&
						++nmembers >= 2)
					{
						result = bms_add_member(result, i);
						break;
					}
				}
			}
		}
	}

	return result;
}

/*
 * build_base_rel_uniquekeys
 *		Build the UniqueKeys of all base relations, if not already done.
 *
 * We do this the first time UniqueKeys are needed, so that queries that
 * never need them don't pay for it.  Also set root->no_uniquekey_relids.
 */
static void
build_base_rel_uniquekeys(PlannerInfo *root)
{
	MemoryContext oldcontext;
	Relids		no_uniquekey_relids = NULL;
	Relids		full_join_relids = NULL;
	int			rti;

	if (root->uniquekeys_built)
		return;

	/* UniqueKeys refer to ECs by index, so the EC list must be final */
	Assert(root->ec_merging_done);

	/* Be sure to keep the results in the planner_cxt even if within GEQO */
	oldcontext = MemoryContextSwitchTo(root->planner_cxt);

	/* The UniqueKeys can only use ECs that exist, so do this first */
	if (root->outer_join_rels != NULL)
		build_base_eclasses(root);

	/* Note the full joins, for use below */
	foreach_node(SpecialJoinInfo, sjinfo, root->join_info_list)
	{
		if (sjinfo->jointype == JOIN_FULL)
			full_join_relids = bms_add_member(full_join_relids,
											  sjinfo->ojrelid);
	}

	for (rti = 1; rti < root->simple_rel_array_size; rti++)
	{
		RelOptInfo *rel = root->simple_rel_array[rti];

		/* there may be empty slots corresponding to non-baserel RTEs */
		if (rel == NULL)
			continue;

		/* Currently we only know about plain relations ... */
		if (rel->reloptkind == RELOPT_BASEREL &&
			rel->rtekind == RTE_RELATION)
		{
			build_plain_rel_uniquekeys(root, rel);

			/* full_nullable_rel_is_marked() needs this for such rels only */
			if (rel->uniquekeys != NIL &&
				bms_overlap(rel->nulling_relids, full_join_relids))
				build_notnull_eclass_indexes(root, rel);
		}

		/* ... and subqueries, though not the parents of appendrels */
		if (rel->reloptkind == RELOPT_BASEREL &&
			rel->rtekind == RTE_SUBQUERY &&
			!root->simple_rte_array[rti]->inh)
			build_subquery_rel_uniquekeys(root, rel);

		if (rel->uniquekeys == NIL)
			no_uniquekey_relids = bms_add_member(no_uniquekey_relids, rti);
	}

	root->no_uniquekey_relids = no_uniquekey_relids;
	root->uniquekeys_built = true;

	MemoryContextSwitchTo(oldcontext);
}

/*
 * build_base_eclasses
 *		Make sure that the un-nulled form of each expression that is nulled
 *		by outer joins has an EC.
 *
 * A column that is only referenced above an outer join that nulls it, say in
 * DISTINCT or in the clause of an upper join, is only in the EC of its nulled
 * form.  UniqueKeys need the EC of the column itself, so create that if it
 * doesn't exist yet.
 */
static void
build_base_eclasses(PlannerInfo *root)
{
	int			nclasses = list_length(root->eq_classes);

	/* ECs created here are appended to the list; no need to visit them */
	for (int i = 0; i < nclasses; i++)
	{
		EquivalenceClass *ec = list_nth_node(EquivalenceClass,
											 root->eq_classes, i);
		EquivalenceMember *em;
		RelOptInfo *rel;
		Expr	   *expr;

		if (ec->ec_has_volatile ||
			!bms_overlap(ec->ec_relids, root->outer_join_rels))
			continue;

		/* We can only do this for an EC that is a single expression */
		if (list_length(ec->ec_members) != 1)
			continue;
		em = linitial_node(EquivalenceMember, ec->ec_members);

		expr = get_base_expr(root, em->em_expr, &rel);
		if (expr == NULL)
			continue;

		/* This finds the EC if it exists, else creates it */
		(void) get_eclass_for_sort_expr(root, expr, ec->ec_opfamilies,
										em->em_datatype, ec->ec_collation,
										0, NULL, true);
	}
}

/*
 * build_plain_rel_uniquekeys
 *		Build UniqueKeys for a plain relation from its unique indexes.
 *
 * An index column that is in no EC cannot be matched by any clause, so an
 * index having such a column yields no UniqueKey.  The exception is a column
 * that a subquery outputs: the outer query can ask about it (see
 * build_subquery_scanjoin_uniquekeys), so we make an EC for it.
 */
static void
build_plain_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel)
{
	List	   *nonnullable_vars = NIL;
	bool		nonnullable_vars_valid = false;

	/* Short-circuit if no EC mentions this relation, except in a subquery */
	if (rel->eclass_indexes == NULL && root->parent_root == NULL)
		return;

	foreach_node(IndexOptInfo, ind, rel->indexlist)
	{
		Bitmapset  *key_ecs = NULL;
		bool		null_aware = true;
		int			c;

		/*
		 * If the index is not unique, or not immediately enforced, or if it's
		 * a partial index, it's useless here.
		 */
		if (!ind->unique || !ind->immediate || ind->indpred != NIL)
			continue;

		for (c = 0; c < ind->nkeycolumns; c++)
		{
			TargetEntry *tle = list_nth_node(TargetEntry, ind->indextlist, c);
			int			attno = ind->indexkeys[c];
			EquivalenceClass *ec;
			List	   *opfamilies;
			Oid			eq_op;
			int			ec_index;

			/*
			 * An EC's opfamily list comes from its equality operator, which
			 * can belong to more opfamilies than just the index's.  So look
			 * up the index column's equality operator to compare with.
			 */
			eq_op = get_opfamily_member_for_cmptype(ind->opfamily[c],
													ind->opcintype[c],
													ind->opcintype[c],
													COMPARE_EQ);
			if (!OidIsValid(eq_op))
				break;

			opfamilies = get_mergejoin_opfamilies(eq_op);
			ec_index = find_ec_index(root, rel, tle->expr, opfamilies,
									 ind->indexcollations[c]);

			/* In a subquery, make an EC for a column that is only output */
			if (ec_index < 0 && root->parent_root != NULL && attno > 0 &&
				bms_is_member(0, rel->attr_needed[attno - rel->min_attr]))
			{
				(void) get_eclass_for_sort_expr(root, tle->expr, opfamilies,
												ind->opcintype[c],
												ind->indexcollations[c],
												0, NULL, true);
				ec_index = find_ec_index(root, rel, tle->expr, opfamilies,
										 ind->indexcollations[c]);
			}
			if (ec_index < 0)
				break;

			/*
			 * A column that is equal to a constant has the same value in all
			 * the relation's rows, so the remaining columns are unique
			 * without it.
			 */
			ec = list_nth_node(EquivalenceClass, root->eq_classes, ec_index);
			if (ec->ec_has_const)
				continue;

			key_ecs = bms_add_member(key_ecs, ec_index);

			/*
			 * Unless the index is NULLS NOT DISTINCT, it allows duplicates
			 * among rows that have a NULL in this column.  So the UniqueKey
			 * is not null_aware unless the column cannot be NULL, due to a
			 * NOT NULL constraint or a strict restriction clause.
			 */
			if (null_aware && !ind->nullsnotdistinct)
			{
				if (attno == 0)
					null_aware = false;
				else if (!bms_is_member(attno, rel->notnullattnums))
				{
					if (!nonnullable_vars_valid)
					{
						foreach_node(RestrictInfo, rinfo, rel->baserestrictinfo)
						{
							nonnullable_vars =
								mbms_add_members(nonnullable_vars,
												 find_nonnullable_vars((Node *) rinfo->clause));
						}
						nonnullable_vars_valid = true;
					}

					if (!mbms_is_member(rel->relid,
										attno - FirstLowInvalidHeapAttributeNumber,
										nonnullable_vars))
						null_aware = false;
				}
			}
		}

		/* Add the UniqueKey if we matched all the index columns */
		if (c == ind->nkeycolumns)
			add_uniquekey(rel, key_ecs, null_aware);
	}
}

/*
 * build_subquery_rel_uniquekeys
 *		Build UniqueKeys for a subquery relation.
 *
 * A subquery can be distinct by construction, through DISTINCT, GROUP BY, a
 * set operation and so on.  query_is_distinct_for() knows the rules, but it
 * answers for a given set of output columns.  So we ask it about each set of
 * columns that could come out distinct that way, and make a UniqueKey of each
 * set for which it says yes.
 *
 * Those constructs consider NULLs equal to each other, so the UniqueKeys are
 * null_aware.
 *
 * A subquery can also be distinct because of the relations it is made of;
 * see build_subquery_scanjoin_uniquekeys().
 */
static void
build_subquery_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel)
{
	Query	   *subquery = root->simple_rte_array[rel->relid]->subquery;
	List	   *distinct_cols = NIL;
	List	   *col_ecs = NIL;
	List	   *const_cols = NIL;
	Bitmapset  *colnos;
	int			i = -1;

	/*
	 * Collect the output columns that are in ECs, in the form that
	 * query_is_distinct_for() wants, along with the ECs they are in.  A
	 * column that is equal to a constant has the same value in all the rows
	 * of interest, so it can be given in every question and needs no place in
	 * a UniqueKey.
	 */
	while ((i = bms_next_member(rel->eclass_indexes, i)) >= 0)
	{
		EquivalenceClass *ec = list_nth_node(EquivalenceClass,
											 root->eq_classes, i);

		/* The members of a broken EC are not known to be equal */
		if (ec->ec_broken || ec->ec_has_volatile)
			continue;

		foreach_node(EquivalenceMember, em, ec->ec_members)
		{
			Var		   *var = (Var *) em->em_expr;
			DistinctColInfo *dcinfo;
			Oid			eq_op;

			/* We may ignore any RelabelType node above the Var */
			if (IsA(var, RelabelType))
				var = (Var *) ((RelabelType *) var)->arg;

			if (!IsA(var, Var) ||
				var->varno != rel->relid ||
				var->varnullingrels != NULL)
				continue;

			eq_op = get_opfamily_member_for_cmptype(linitial_oid(ec->ec_opfamilies),
													em->em_datatype,
													em->em_datatype,
													COMPARE_EQ);
			if (!OidIsValid(eq_op))
				continue;

			dcinfo = palloc_object(DistinctColInfo);
			dcinfo->colno = var->varattno;
			dcinfo->opid = eq_op;
			dcinfo->collid = ec->ec_collation;

			if (ec->ec_has_const)
				const_cols = lappend(const_cols, dcinfo);
			else
			{
				distinct_cols = lappend(distinct_cols, dcinfo);
				col_ecs = lappend_int(col_ecs, i);
			}
		}
	}

	/* Build the UniqueKeys that follow from the subquery's own relations */
	build_subquery_scanjoin_uniquekeys(root, rel, distinct_cols, col_ecs,
									   const_cols);

	if (!query_supports_distinctness(subquery))
		return;

	/* A subquery that returns at most one row needs no columns at all */
	if (add_subquery_uniquekey(rel, subquery, distinct_cols, col_ecs,
							   const_cols, NULL))
		return;

	/* The DISTINCT columns */
	colnos = NULL;
	foreach_node(SortGroupClause, sgc, subquery->distinctClause)
	{
		TargetEntry *tle = get_sortgroupclause_tle(sgc, subquery->targetList);

		colnos = bms_add_member(colnos, tle->resno);
	}
	if (colnos != NULL)
		add_subquery_uniquekey(rel, subquery, distinct_cols, col_ecs,
							   const_cols, colnos);

	/* The GROUP BY columns */
	colnos = NULL;
	foreach_node(SortGroupClause, sgc, subquery->groupClause)
	{
		TargetEntry *tle = get_sortgroupclause_tle(sgc, subquery->targetList);

		colnos = bms_add_member(colnos, tle->resno);
	}
	if (colnos != NULL)
		add_subquery_uniquekey(rel, subquery, distinct_cols, col_ecs,
							   const_cols, colnos);

	/* All the output columns, for a set operation */
	if (subquery->setOperations)
	{
		colnos = NULL;
		foreach_node(TargetEntry, tle, subquery->targetList)
		{
			if (!tle->resjunk)
				colnos = bms_add_member(colnos, tle->resno);
		}
		add_subquery_uniquekey(rel, subquery, distinct_cols, col_ecs,
							   const_cols, colnos);
	}
}

/*
 * add_subquery_uniquekey
 *		Subroutine for build_subquery_rel_uniquekeys: add a UniqueKey for the
 *		given output columns if the subquery is distinct on them.
 *
 * distinct_cols and col_ecs list the columns that can be used and the ECs
 * they are in.  const_cols lists columns that are equal to constants, which
 * are given in addition.  Returns true if the subquery is distinct on the
 * given columns.
 */
static bool
add_subquery_uniquekey(RelOptInfo *rel, Query *subquery,
					   List *distinct_cols, List *col_ecs,
					   List *const_cols, Bitmapset *colnos)
{
	List	   *cols = list_copy(const_cols);
	Bitmapset  *key_ecs = NULL;
	ListCell   *lc1;
	ListCell   *lc2;

	forboth(lc1, distinct_cols, lc2, col_ecs)
	{
		DistinctColInfo *dcinfo = lfirst(lc1);

		if (bms_is_member(dcinfo->colno, colnos))
		{
			cols = lappend(cols, dcinfo);
			key_ecs = bms_add_member(key_ecs, lfirst_int(lc2));
		}
	}

	if (!query_is_distinct_for(subquery, cols))
		return false;

	add_uniquekey(rel, key_ecs, true);
	return true;
}

/*
 * build_subquery_scanjoin_uniquekeys
 *		Build UniqueKeys for a subquery relation from the UniqueKeys of the
 *		subquery's own base relations.
 *
 * Each row that the subquery emits comes from a row of its scan/join
 * relation, or from a group of such rows that agree on the output columns
 * that are not aggregates.  So the subquery is distinct on a set of output
 * columns if its scan/join relation is.  That holds whatever else the
 * subquery does, except for set-returning functions in its targetlist and
 * grouping sets, which can emit a row more than once.
 *
 * There is no telling up front which sets of columns the outer query will
 * ask about.  The useful ones are those that cover a UniqueKey of one of the
 * subquery's base relations, so we try each of those UniqueKeys whose columns
 * are all output: if the subquery's scan/join relation is distinct for it,
 * the ECs of those output columns make a UniqueKey.  If none of them will do
 * by itself, we try all of them together.
 *
 * The parameters are as for add_subquery_uniquekey().
 */
static void
build_subquery_scanjoin_uniquekeys(PlannerInfo *root, RelOptInfo *rel,
								   List *distinct_cols, List *col_ecs,
								   List *const_cols)
{
	PlannerInfo *subroot = rel->subroot;
	RelOptInfo *scanjoin_rel;
	List	   *sub_ecs = NIL;
	Bitmapset  *const_sub_ecs = NULL;
	Bitmapset  *all_key_ecs = NULL;
	Bitmapset  *all_sub_ecs = NULL;
	int			ncandidates = 0;
	bool		found = false;
	int			relid;

	if (subroot == NULL ||
		subroot->parse->setOperations ||
		subroot->parse->groupingSets ||
		subroot->parse->hasTargetSRFs)
		return;

	/* Find the subquery's scan/join relation */
	if (bms_get_singleton_member(subroot->all_query_rels, &relid))
		scanjoin_rel = find_base_rel_noerr(subroot, relid);
	else
		scanjoin_rel = find_join_rel(subroot, subroot->all_query_rels);
	if (scanjoin_rel == NULL)
		return;

	build_base_rel_uniquekeys(subroot);

	/*
	 * Find the subquery's EC for each of the columns.  Those of columns that
	 * the outer query equates to constants are known in every question.
	 */
	foreach_ptr(DistinctColInfo, dcinfo, distinct_cols)
	{
		sub_ecs = lappend_int(sub_ecs, find_output_ec_index(subroot, dcinfo));
	}
	foreach_ptr(DistinctColInfo, dcinfo, const_cols)
	{
		int			sub_ec = find_output_ec_index(subroot, dcinfo);

		if (sub_ec >= 0)
			const_sub_ecs = bms_add_member(const_sub_ecs, sub_ec);
	}

	relid = -1;
	while ((relid = bms_next_member(scanjoin_rel->relids, relid)) >= 0)
	{
		RelOptInfo *subrel = subroot->simple_rel_array[relid];

		/* ignore the relids of outer joins */
		if (subrel == NULL)
			continue;

		foreach_node(UniqueKey, ukey, subrel->uniquekeys)
		{
			Bitmapset  *key_ecs = NULL;
			Bitmapset  *known_ecs;
			int			i = -1;

			/* Translate the UniqueKey to the ECs of the outer query */
			while ((i = bms_next_member(ukey->eclass_indexes, i)) >= 0)
			{
				ListCell   *lc1;
				ListCell   *lc2;

				if (bms_is_member(i, const_sub_ecs))
					continue;

				forboth(lc1, sub_ecs, lc2, col_ecs)
				{
					if (lfirst_int(lc1) == i)
						break;
				}
				if (lc1 == NULL)
					break;
				key_ecs = bms_add_member(key_ecs, lfirst_int(lc2));
			}

			/* Skip the UniqueKey if one of its columns is not output */
			if (i >= 0)
				continue;

			known_ecs = bms_union(ukey->eclass_indexes, const_sub_ecs);

			if (rel_is_distinct_for_ecs(subroot, scanjoin_rel, known_ecs, true))
			{
				add_uniquekey(rel, key_ecs, true);
				found = true;
			}
			else if (rel_is_distinct_for_ecs(subroot, scanjoin_rel, known_ecs,
											 false))
			{
				add_uniquekey(rel, key_ecs, false);
				found = true;
			}

			all_key_ecs = bms_add_members(all_key_ecs, key_ecs);
			all_sub_ecs = bms_add_members(all_sub_ecs, known_ecs);
			ncandidates++;
		}
	}

	/* If no single UniqueKey would do, try them all together */
	if (!found && ncandidates > 1)
	{
		if (rel_is_distinct_for_ecs(subroot, scanjoin_rel, all_sub_ecs, true))
			add_uniquekey(rel, all_key_ecs, true);
		else if (rel_is_distinct_for_ecs(subroot, scanjoin_rel, all_sub_ecs,
										 false))
			add_uniquekey(rel, all_key_ecs, false);
	}
}

/*
 * find_output_ec_index
 *		Find the EC that the given output column of the query is in, for use
 *		by the outer query.
 *
 * For a column that is nulled by outer joins, this is the EC of its
 * un-nulled form, as in add_base_ec_index().  Returns -1 if there is none.
 */
static int
find_output_ec_index(PlannerInfo *root, DistinctColInfo *dcinfo)
{
	TargetEntry *tle = get_tle_by_resno(root->processed_tlist, dcinfo->colno);
	RelOptInfo *rel;
	Expr	   *expr;

	if (tle == NULL)
		return -1;

	expr = get_output_base_expr(root, tle, &rel);
	if (expr == NULL)
		return -1;

	return find_ec_index(root, rel, expr,
						 get_mergejoin_opfamilies(dcinfo->opid),
						 dcinfo->collid);
}

/*
 * build_notnull_eclass_indexes
 *		Find the ECs that have a member that is a column of the relation that
 *		cannot be NULL, and save them in rel->notnull_eclass_indexes.
 *
 * A column cannot be NULL if it has a NOT NULL constraint or if a strict
 * restriction clause of the relation filters out the NULLs.
 */
static void
build_notnull_eclass_indexes(PlannerInfo *root, RelOptInfo *rel)
{
	List	   *nonnullable_vars = NIL;
	Bitmapset  *result = NULL;
	int			i = -1;

	foreach_node(RestrictInfo, rinfo, rel->baserestrictinfo)
	{
		nonnullable_vars =
			mbms_add_members(nonnullable_vars,
							 find_nonnullable_vars((Node *) rinfo->clause));
	}

	while ((i = bms_next_member(rel->eclass_indexes, i)) >= 0)
	{
		EquivalenceClass *ec = list_nth_node(EquivalenceClass,
											 root->eq_classes, i);

		foreach_node(EquivalenceMember, em, ec->ec_members)
		{
			Var		   *var = (Var *) em->em_expr;

			if (!IsA(var, Var) ||
				var->varno != rel->relid ||
				var->varnullingrels != NULL)
				continue;

			if ((var->varattno > 0 &&
				 bms_is_member(var->varattno, rel->notnullattnums)) ||
				mbms_is_member(var->varno,
							   var->varattno - FirstLowInvalidHeapAttributeNumber,
							   nonnullable_vars))
			{
				result = bms_add_member(result, i);
				break;
			}
		}
	}

	rel->notnull_eclass_indexes = result;
}

/*
 * add_uniquekey
 *		Add a UniqueKey for the given set of ECs to the relation, unless it
 *		is redundant.
 *
 * A UniqueKey is redundant if another one of the relation is a subset of it
 * and can be used wherever it can.
 */
static void
add_uniquekey(RelOptInfo *rel, Bitmapset *eclass_indexes,
			  bool null_aware)
{
	UniqueKey  *ukey;
	ListCell   *lc;

	foreach(lc, rel->uniquekeys)
	{
		UniqueKey  *other = lfirst_node(UniqueKey, lc);

		/* If an existing key makes the new one redundant, we're done */
		if (bms_is_subset(other->eclass_indexes, eclass_indexes) &&
			(other->null_aware || !null_aware))
			return;

		/* Remove any existing key that the new one makes redundant */
		if (bms_is_subset(eclass_indexes, other->eclass_indexes) &&
			(null_aware || !other->null_aware))
			rel->uniquekeys = foreach_delete_current(rel->uniquekeys, lc);
	}

	ukey = makeNode(UniqueKey);
	ukey->eclass_indexes = eclass_indexes;
	ukey->null_aware = null_aware;
	rel->uniquekeys = lappend(rel->uniquekeys, ukey);
}

/*
 * find_ec_index
 *		Find the EC that has the given expression of the given base relation
 *		as a member, and that uses the given opfamilies and collation.
 *
 * Returns the EC's index in root->eq_classes, or -1 if there is none.
 */
static int
find_ec_index(PlannerInfo *root, RelOptInfo *rel, Expr *expr,
			  List *opfamilies, Oid collation)
{
	int			i = -1;

	while ((i = bms_next_member(rel->eclass_indexes, i)) >= 0)
	{
		EquivalenceClass *ec = list_nth_node(EquivalenceClass,
											 root->eq_classes, i);

		/* The members of a broken EC are not known to be equal */
		if (ec->ec_broken || ec->ec_has_volatile)
			continue;

		if (ec->ec_collation != collation ||
			!equal(ec->ec_opfamilies, opfamilies))
			continue;

		if (find_ec_member_matching_expr(ec, expr, NULL))
			return i;
	}

	return -1;
}

/*
 * add_base_ec_index
 *		Add the index of the given EC to the given set.  If it is the EC of
 *		an expression that is nulled by outer joins, add the index of the EC
 *		of the un-nulled expression instead.
 *
 * Wherever the nulled expression is not NULL, it is equal to the un-nulled
 * one.  Where it is NULL, either the un-nulled one is NULL too or the
 * relation it belongs to is null-extended; joinrel_is_distinct_for_ecs()
 * sees to that when it matters.
 *
 * Nothing is added if the un-nulled expression cannot be identified or has
 * no EC.  The caller then has less to work with, which is always safe.
 */
static Bitmapset *
add_base_ec_index(PlannerInfo *root, Bitmapset *eclass_indexes,
				  EquivalenceClass *ec)
{
	EquivalenceMember *em;
	RelOptInfo *rel;
	Expr	   *expr;
	int			ec_index;

	if (ec == NULL)
		return eclass_indexes;

	/* the given EC might be non-canonical, so chase up to the top */
	while (ec->ec_merged)
		ec = ec->ec_merged;

	if (!bms_overlap(ec->ec_relids, root->outer_join_rels))
		return bms_add_member(eclass_indexes, ec->ec_index);

	/* We can only do this for an EC that is a single expression */
	if (list_length(ec->ec_members) != 1)
		return eclass_indexes;
	em = linitial_node(EquivalenceMember, ec->ec_members);

	expr = get_base_expr(root, em->em_expr, &rel);
	if (expr == NULL)
		return eclass_indexes;

	ec_index = find_ec_index(root, rel, expr,
							 ec->ec_opfamilies, ec->ec_collation);
	if (ec_index < 0)
		return eclass_indexes;

	return bms_add_member(eclass_indexes, ec_index);
}

/*
 * get_base_expr
 *		Get the un-nulled form of the given expression, which is the
 *		expression itself if it is not nulled by any outer join.  Also return
 *		the base relation the expression belongs to in *rel.
 *
 * Returns NULL if the expression does not belong to a single base relation.
 * Also returns NULL if the expression is nulled by outer joins but is not
 * certain to be NULL when the relation is null-extended, in which case the
 * un-nulled form doesn't stand for it.  A Var is always NULL then, and so is
 * a strict expression.
 */
static Expr *
get_base_expr(PlannerInfo *root, Expr *expr, RelOptInfo **rel)
{
	Relids		relids = pull_varnos(root, (Node *) expr);
	Expr	   *arg = expr;
	int			relid;

	if (!bms_get_singleton_member(bms_difference(relids,
												 root->outer_join_rels),
								  &relid))
		return NULL;
	*rel = find_base_rel_noerr(root, relid);
	if (*rel == NULL)
		return NULL;

	if (!bms_overlap(relids, root->outer_join_rels))
		return expr;

	while (IsA(arg, RelabelType))
		arg = ((RelabelType *) arg)->arg;
	if (!IsA(arg, Var) &&
		contain_nonstrict_functions((Node *) expr))
		return NULL;

	return (Expr *) remove_nulling_relids((Node *) expr,
										  root->outer_join_rels, NULL);
}

/*
 * get_output_base_expr
 *		As get_base_expr(), for an output column of the query.
 *
 * Returns NULL for a junk column, and for one that contains aggregates or
 * window functions: such a column is not computed from a row of a single
 * base relation.
 */
static Expr *
get_output_base_expr(PlannerInfo *root, TargetEntry *tle, RelOptInfo **rel)
{
	if (tle->resjunk ||
		contain_agg_clause((Node *) tle->expr) ||
		contain_window_function((Node *) tle->expr))
		return NULL;

	return get_base_expr(root, tle->expr, rel);
}

/*
 * rel_has_uniquekey_in
 *		Does the relation have a UniqueKey that is a subset of the given set
 *		of ECs?
 *
 * If nulls_equal is true, the values of the ECs can be NULL, so only
 * UniqueKeys that are null_aware are of use.  The exception is a UniqueKey
 * whose ECs are all in joined_ecs, which are those that join clauses keep
 * from being NULL in the rows that the relation contributes to the join
 * relation being considered (see get_joined_ecs).
 */
static bool
rel_has_uniquekey_in(PlannerInfo *root, RelOptInfo *rel,
					 Bitmapset *eclass_indexes, Bitmapset *joined_ecs,
					 bool nulls_equal)
{
	foreach_node(UniqueKey, ukey, rel->uniquekeys)
	{
		bool		broken = false;
		int			i = -1;

		if (nulls_equal && !ukey->null_aware &&
			!bms_is_subset(ukey->eclass_indexes, joined_ecs))
			continue;

		if (!bms_is_subset(ukey->eclass_indexes, eclass_indexes))
			continue;

		/* An EC can be found to be broken after the key was built */
		while ((i = bms_next_member(ukey->eclass_indexes, i)) >= 0)
		{
			EquivalenceClass *ec = list_nth_node(EquivalenceClass,
												 root->eq_classes, i);

			if (ec->ec_broken)
			{
				broken = true;
				break;
			}
		}

		if (!broken)
			return true;
	}

	return false;
}

/*
 * full_nullable_rel_is_marked
 *		Does one of the given ECs tell whether the relation, which is nulled
 *		by a full join within the join relation being considered, is
 *		null-extended?
 *
 * Showing that a relation has only one row of its own for the given ECs is
 * not enough under a full join, as nothing else decides whether the relation
 * is represented by that row or by a null-extended row.  (Under a left join,
 * the left-hand row decides that: the join emits either the matching
 * right-hand rows or a single null-extended row, never both.)  So we require
 * one of the given ECs to tell the two apart.  If nulls_equal is false, any
 * EC with a member of the relation does, since rows that are NULL in it are
 * ignored.  Otherwise it must have a column of the relation that cannot be
 * NULL, because a null-extended row looks just like a row of the relation's
 * own that has NULLs in all the given ECs.
 */
static bool
full_nullable_rel_is_marked(PlannerInfo *root, RelOptInfo *rel,
							Bitmapset *eclass_indexes, bool nulls_equal)
{
	if (nulls_equal)
		return rel_has_notnull_column_in(root, rel, eclass_indexes);

	return bms_overlap(get_own_eclass_indexes(root, rel), eclass_indexes);
}

/*
 * rel_has_notnull_column_in
 *		Does any of the given ECs have a member that is a column of the
 *		relation that cannot be NULL?
 *
 * This relies on rel->notnull_eclass_indexes, which is only set up for
 * relations that are nulled by a full join.
 */
static bool
rel_has_notnull_column_in(PlannerInfo *root, RelOptInfo *rel,
						  Bitmapset *eclass_indexes)
{
	int			i = -1;

	while ((i = bms_next_member(rel->notnull_eclass_indexes, i)) >= 0)
	{
		EquivalenceClass *ec;

		if (!bms_is_member(i, eclass_indexes))
			continue;

		/* An EC can be found to be broken after the set was built */
		ec = list_nth_node(EquivalenceClass, root->eq_classes, i);
		if (!ec->ec_broken)
			return true;
	}

	return false;
}

/*
 * get_own_eclass_indexes
 *		Get the indexes of the ECs that have a member belonging to the given
 *		base relation alone.
 *
 * This is not the same as rel->eclass_indexes, which also includes ECs whose
 * members only use the relation together with other relations.
 *
 * The result is cached in the RelOptInfo.  An empty result is not cached,
 * but that is cheap to recompute.
 */
static Bitmapset *
get_own_eclass_indexes(PlannerInfo *root, RelOptInfo *rel)
{
	MemoryContext oldcontext;
	Bitmapset  *result = NULL;
	int			i = -1;

	if (rel->own_eclass_indexes != NULL)
		return rel->own_eclass_indexes;

	/* Be sure to keep the result in the planner_cxt even if within GEQO */
	oldcontext = MemoryContextSwitchTo(root->planner_cxt);

	while ((i = bms_next_member(rel->eclass_indexes, i)) >= 0)
	{
		EquivalenceClass *ec = list_nth_node(EquivalenceClass,
											 root->eq_classes, i);

		foreach_node(EquivalenceMember, em, ec->ec_members)
		{
			if (bms_equal(em->em_relids, rel->relids))
			{
				result = bms_add_member(result, i);
				break;
			}
		}
	}

	MemoryContextSwitchTo(oldcontext);

	rel->own_eclass_indexes = result;

	return result;
}
