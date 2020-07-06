/**
 * Author: Severin Gsponer (svgsponer@gmail.com)
 *
 * SNode: represents a node in a searchtree for SEQL
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

#include "SNode.h"

// Shrink the list of total occurrences to contain just support doc_ids, instead
// of doc_ids and occurences.
void SNode::shrink() {
    std::vector<int> tmp;

    for (auto const &currLoc : loc) {
        if (currLoc < 0) {
            tmp.push_back(currLoc);
        }
    }
    loc = std::move(tmp);
    loc.shrink_to_fit();

    // Does not shrink the capacity of the location erase remove idome
    // loc.erase( std::remove_if(loc.begin(), loc.end(), [](int i){return i >= // 0;}),loc.end());
    last_docid = -1;
}

// Return the support of current ngram.
// Simply count the negative loc as doc_ids.
unsigned int SNode::support() const {
    return std::count_if(begin(loc), end(loc),
                         [](int currLoc) { return currLoc < 0; });
}

std::vector<int> SNode::getLoc() { return loc; }

std::string SNode::getNgram() {
    std::string ngram = "";
    if (!tokenType) { // If word-level token: a bb cd a bb
        for (SNode *t = this; t!=nullptr; t = t->prev) {
            ngram = " " + t->ne + ngram;
        }
        // skip the space in front of the ngram
        ngram.assign(ngram.substr(1));

    } else { // char-level tokens: abbcdabb
        for (SNode *t = this; t!=nullptr; t = t->prev) {
            ngram = t->ne + ngram;
        }
    }
    return ngram;
}


bool SNode::violateWildcardConstraint() {
    int numberOfWildcards = 0;
    int numberOfConsecWildcards = 0 ;

    for (SNode *t = this; t != nullptr; t = t->prev) {
        if (t->ne.compare("*") == 0) {
            numberOfWildcards++;
            numberOfConsecWildcards++;
            if (numberOfWildcards > totalWildcardLimit) {
                return true;
            }
        }else{
            if (numberOfConsecWildcards > consecWildcardLimit){
                return true;
            }
            numberOfConsecWildcards = 0;
        }
    }
    return false;
}

void SNode::setupWildcardConstraint(int _totalWildcardLimit,
                                    int _consecWildcardLimit) {
    if (_totalWildcardLimit == 0) {
        if (_consecWildcardLimit == 0) {
            hasWildcardConstraints = false;
        } else {
            hasWildcardConstraints = true;
            consecWildcardLimit = _consecWildcardLimit;
            totalWildcardLimit = std::numeric_limits<int>::max();
        }
    }else{
        hasWildcardConstraints = true;
        if(_consecWildcardLimit == 0 || _consecWildcardLimit > _totalWildcardLimit) {
            totalWildcardLimit = _totalWildcardLimit;
            consecWildcardLimit = totalWildcardLimit;
        }
        else{
            totalWildcardLimit = _totalWildcardLimit;
            consecWildcardLimit = _consecWildcardLimit;
        }
    }
}
// Add a doc_id and position of occurrence to the total list of occurrences,
// for this ngram.
// Encode the doc_id in the vector of locations.
// Negative entry means new doc_id.
void SNode::add(unsigned int docid, int pos) {
    if (last_docid != static_cast<int>(docid)) {
        loc.push_back(-static_cast<int>(docid + 1));
    }
    loc.push_back(pos);
    last_docid = static_cast<int>(docid);
}

SNode::~SNode(){
	for (SNode* child:next){
		delete child;
	}
    // std::cout << "Deconstr"<< std::endl;
};
bool SNode::tokenType = true;
bool SNode::hasWildcardConstraints = true;
int SNode::totalWildcardLimit = 0;
int SNode::consecWildcardLimit = 0;
