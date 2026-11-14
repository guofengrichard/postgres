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
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "rewrite/rewriteManip.h"
#include "utils/lsyscache.h"

static bool joinrel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *joinrel,
										Bitmapset *known_ecs,
										bool nulls_equal);
static Bitmapset *get_left_join_ecs(PlannerInfo *root, RelOptInfo *joinrel);
static Bitmapset *get_joined_ecs(PlannerInfo *root, RelOptInfo *joinrel,
								 Bitmapset *left_join_ecs);
static void build_base_rel_uniquekeys(PlannerInfo *root);
static void build_base_eclasses(PlannerInfo *root);
static void build_plain_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel);
static void build_notnull_eclass_indexes(PlannerInfo *root, RelOptInfo *rel);
static void add_uniquekey(RelOptInfo *rel, Bitmapset *eclass_indexes,
						  bool null_aware);
static int	find_ec_index(PlannerInfo *root, RelOptInfo *rel, Expr *expr,
						  List *opfamilies, Oid collation);
static Bitmapset *add_base_ec_index(PlannerInfo *root,
									Bitmapset *eclass_indexes,
									EquivalenceClass *ec);
static Expr *get_base_expr(PlannerInfo *root, EquivalenceClass *ec,
						   RelOptInfo **rel);
static bool rel_has_uniquekey_in(PlannerInfo *root, RelOptInfo *rel,
								 Bitmapset *eclass_indexes,
								 Bitmapset *joined_ecs,
								 bool nulls_equal, bool full_nullable);
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
	Assert(IS_JOIN_REL(joinrel));

	/* A grouped join rel's rows are not combinations of base rel rows */
	if (IS_GROUPED_REL(joinrel))
		return false;

	build_base_rel_uniquekeys(root);

	/* Every base rel in the join must have a UniqueKey */
	return !bms_overlap(joinrel->relids, root->no_uniquekey_relids);
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
 * rel_is_distinct_for_pathkeys
 *		Does the relation return only distinct rows according to the given
 *		pathkeys?
 *
 * Return true if no two rows emitted by the relation could possibly have
 * equal values in all the pathkeys' ECs, with NULLs considered equal to each
 * other.  The sort ordering of the pathkeys is of no interest.
 *
 * An empty list means all rows are considered equal, so the relation is then
 * distinct only if it has at most one row.
 */
bool
rel_is_distinct_for_pathkeys(PlannerInfo *root, RelOptInfo *rel,
							 List *pathkeys)
{
	Bitmapset  *known_ecs = NULL;

	if (IS_JOIN_REL(rel))
	{
		if (!joinrel_supports_distinctness(root, rel))
			return false;
	}
	else if (rel->reloptkind == RELOPT_BASEREL)
		build_base_rel_uniquekeys(root);
	else
		return false;			/* we know nothing about other relations */

	foreach_node(PathKey, pathkey, pathkeys)
	{
		known_ecs = add_base_ec_index(root, known_ecs, pathkey->pk_eclass);
	}

	if (IS_JOIN_REL(rel))
		return joinrel_is_distinct_for_ecs(root, rel, known_ecs, true);

	return rel_has_uniquekey_in(root, rel, known_ecs, NULL, true, false);
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
 * The caller must have checked joinrel_supports_distinctness().
 */
static bool
joinrel_is_distinct_for_ecs(PlannerInfo *root, RelOptInfo *joinrel,
							Bitmapset *known_ecs, bool nulls_equal)
{
	Bitmapset  *left_join_ecs;
	Bitmapset  *joined_ecs = NULL;
	Relids		full_join_relids = NULL;
	Relids		remaining;
	bool		found;

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
	 * rel_has_uniquekey_in().  So find the full joins performed within the
	 * join relation.
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

	/* The relids of outer joins are not relations to account for */
	remaining = bms_difference(joinrel->relids, root->outer_join_rels);

	do
	{
		int			relid = -1;

		found = false;

		while ((relid = bms_next_member(remaining, relid)) >= 0)
		{
			RelOptInfo *rel = find_base_rel(root, relid);
			bool		full_nullable;

			/* Is the rel nulled by a full join within this join relation? */
			full_nullable = bms_overlap(rel->nulling_relids, full_join_relids);

			if (!rel_has_uniquekey_in(root, rel, known_ecs, joined_ecs,
									  nulls_equal, full_nullable))
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

	/*
	 * We don't handle semijoins and antijoins.  Rather than checking for them
	 * in each join relation, treat their right-hand relations as having no
	 * UniqueKeys.  Also note the full joins, for use below.
	 */
	foreach_node(SpecialJoinInfo, sjinfo, root->join_info_list)
	{
		if (sjinfo->jointype == JOIN_SEMI || sjinfo->jointype == JOIN_ANTI)
			no_uniquekey_relids = bms_add_members(no_uniquekey_relids,
												  sjinfo->syn_righthand);
		else if (sjinfo->jointype == JOIN_FULL)
			full_join_relids = bms_add_member(full_join_relids,
											  sjinfo->ojrelid);
	}

	for (rti = 1; rti < root->simple_rel_array_size; rti++)
	{
		RelOptInfo *rel = root->simple_rel_array[rti];

		/* there may be empty slots corresponding to non-baserel RTEs */
		if (rel == NULL)
			continue;

		/* Currently we only know about plain relations */
		if (rel->reloptkind == RELOPT_BASEREL &&
			rel->rtekind == RTE_RELATION)
		{
			build_plain_rel_uniquekeys(root, rel);

			/* rel_has_uniquekey_in() needs this for such relations only */
			if (rel->uniquekeys != NIL &&
				bms_overlap(rel->nulling_relids, full_join_relids))
				build_notnull_eclass_indexes(root, rel);
		}

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

		expr = get_base_expr(root, ec, &rel);
		if (expr == NULL)
			continue;

		/* This finds the EC if it exists, else creates it */
		em = linitial_node(EquivalenceMember, ec->ec_members);
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
 * index having such a column yields no UniqueKey.
 */
static void
build_plain_rel_uniquekeys(PlannerInfo *root, RelOptInfo *rel)
{
	List	   *nonnullable_vars = NIL;
	bool		nonnullable_vars_valid = false;

	/* Short-circuit if no EC mentions this relation */
	if (rel->eclass_indexes == NULL)
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
			EquivalenceClass *ec;
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

			ec_index = find_ec_index(root, rel, tle->expr,
									 get_mergejoin_opfamilies(eq_op),
									 ind->indexcollations[c]);
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
				int			attno = ind->indexkeys[c];

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
 * relation it belongs to is null-extended; rel_has_uniquekey_in() sees to
 * that when it matters.
 *
 * Nothing is added if the un-nulled expression cannot be identified or has
 * no EC.  The caller then has less to work with, which is always safe.
 */
static Bitmapset *
add_base_ec_index(PlannerInfo *root, Bitmapset *eclass_indexes,
				  EquivalenceClass *ec)
{
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

	expr = get_base_expr(root, ec, &rel);
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
 *		Get the un-nulled form of the expression that the given EC, which
 *		involves outer joins, consists of.  Also return the base relation
 *		the expression belongs to in *rel.
 *
 * Returns NULL if the EC is not that simple, or if the expression is not
 * certain to be NULL when the relation is null-extended, in which case the
 * un-nulled form doesn't stand for it.  A Var is always NULL then, and so is
 * a strict expression.
 */
static Expr *
get_base_expr(PlannerInfo *root, EquivalenceClass *ec, RelOptInfo **rel)
{
	EquivalenceMember *em;
	Expr	   *expr;
	Relids		relids;
	int			relid;

	if (list_length(ec->ec_members) != 1)
		return NULL;
	em = linitial_node(EquivalenceMember, ec->ec_members);

	/* The expression must belong to a single base relation */
	relids = bms_difference(em->em_relids, root->outer_join_rels);
	if (!bms_get_singleton_member(relids, &relid))
		return NULL;
	*rel = find_base_rel_noerr(root, relid);
	if (*rel == NULL)
		return NULL;

	expr = em->em_expr;
	while (IsA(expr, RelabelType))
		expr = ((RelabelType *) expr)->arg;
	if (!IsA(expr, Var) &&
		contain_nonstrict_functions((Node *) expr))
		return NULL;

	return (Expr *) remove_nulling_relids((Node *) em->em_expr,
										  root->outer_join_rels, NULL);
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
 *
 * full_nullable tells whether the relation is nulled by a full join within
 * the join relation being considered.  A UniqueKey leaves only one real row
 * of the relation, but under a full join nothing else decides whether the
 * relation is represented by that row or by a null-extended row.  (Under a
 * left join, the left-hand row decides that: the join emits either the
 * matching right-hand rows or a single null-extended row, never both.)  So
 * we require one of the given ECs to tell the two apart.  If nulls_equal is
 * false, any EC with a member of the relation does, since rows that are NULL
 * in it are ignored.  Otherwise it must have a column of the relation that
 * cannot be NULL, because a null-extended row looks just like a row of the
 * relation's own that has NULLs in all the given ECs.
 */
static bool
rel_has_uniquekey_in(PlannerInfo *root, RelOptInfo *rel,
					 Bitmapset *eclass_indexes, Bitmapset *joined_ecs,
					 bool nulls_equal, bool full_nullable)
{
	if (full_nullable &&
		!(nulls_equal ?
		  rel_has_notnull_column_in(root, rel, eclass_indexes) :
		  bms_overlap(get_own_eclass_indexes(root, rel), eclass_indexes)))
		return false;

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
