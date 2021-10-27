/**
 *   \file SNode.h
 *   \brief Class for SNodes in the searchtree for SEQL
 *
 *  Class for SNodes in the search tree of SEQL
 *
 *
 */
#ifndef SNODE_H
#define SNODE_H

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>

class SNode
{
private:
    // Last docid.
    int last_docid;

public:
    static bool tokenType;
    static int totalWildcardLimit;
    static int consecWildcardLimit;
    static bool hasWildcardConstraints;

    // Pointer to previous ngram.
    SNode *prev;

    // Label of SNode where extension is done.
    std::string ne;

    // Total list of occurrences in the entire collection for an
    // ngram. A sort of expanded inverted index.
    std::vector <int> loc;

    // Vector of ngrams which are extensions of current ngram.
    std::vector <SNode *> next;

    // Shrink the list of total occurrences to contain just support
    // doc_ids, instead of doc_ids and occurences.
    void shrink ();

    // Return the support of current ngram.
    // Simply count the negative loc as doc_ids.
    unsigned int support () const;

    // Returns the vector of locations
    std::vector<int> getLoc();

    // Returns the full ngram of which this node represents
    std::string getNgram();

    // Set up of the wildcard constraints
    // there are two types of wildcard constraint:
    // 1. total wildcard limit
    // 2. number of consecutive wildcards
    // The rules for setup are the following:
    // If both are zero => no constraints
    // total limit set but no consecutive limit set => consecutive limit set to total limit
    // consecutive limit set but no total limit set => total limit set to max int.
    // consecutive limit greater than total limit => consecutive limit set to total limit
    static void setupWildcardConstraint(int _totalWildcardLimit, int _consecWildcardLimit);

    // checks if this ngram violates any of the wildcard constraints,
    bool violateWildcardConstraint();

    // Add a doc_id and position of occurrence to the list of occurrences,
    // for this ngram.
    // Encode the doc_id in the vector of locations.
    // Negative entry means new doc_id.
    void add (unsigned int docid, int pos);

    SNode(): last_docid(-1), prev(0) {};
    ~SNode();

};
#endif
