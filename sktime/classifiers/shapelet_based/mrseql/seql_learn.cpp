/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 * SEQL: Sequence Learner
 * This library trains ElasticNet-regularized Logistic Regression and L2-loss (squared-hinge-loss) SVM for Classifying Sequences in the feature space of all possible
 * subsequences in the given training set.
 * Elastic Net regularizer: alpha * L1 + (1 - alpha) * L2, which combines L1 and L2 penalty effects. L1 influences the sparsity of the model, L2 corrects potentially high
 * coeficients resulting due to feature correlation (see Regularization Paths for Generalized Linear Models via Coordinate Descent, by Friedman et al, 2010).
 *
 * The user can influence the outcome classification model by specifying the following parameters:
 * [-o objective] (objective function; choice between logistic regression, squared-hinge-svm and squared error. By default: logistic regression.)
 * [-T maxitr] (number of optimization iterations; by default this is set using a convergence threshold on the aggregated change in score predictions.)
 * [-l minpat] (constraint on the min length of any feature)
 * [-L maxpat] (constraint on the max length of any feature)
 * [-m minsup] (constraint on the min support of any feature, i.e. number of sequences containing the feature)
 * [-g maxgap] (number of total wildcards allowed in a feature, e.g. a**b, is a feature of size 4 with any 2 characters in the middle)
 * [-G maxcongap] (number of consecutive wildcards allowed in a feature, e.g. a**b, is a feature of size 4 with any 2 characters in the middle)
 * [-n token_type] (word or character-level token to allow sequences such as 'ab cd ab' and 'abcdab')
 * [-C regularizer_value] value of the regularization parameter, the higher value means more regularization
 * [-a alpha] (weight on L1 vs L2 regularizer, alpha=0.5 means equal weight for l1 and l2)
 * [-r traversal_strategy] (BFS or DFS traversal of the search space), BFS by default
 * [-c convergence_threshold] (stopping threshold for optimisation iterations based on change in aggregated score predictions)
 * [-v verbosity] (amount of printed detail about the model)
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

/* The obj fct is: loss(x, y, beta) + C * ElasticNetReg(alpha, beta).
 */

#include <cfloat>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <set>
#include <iterator>
#include <cstdlib>
#include <cstring>
// #include <unistd.h>
#include "common.h"
// #include "sys/time.h"
#include <list>
#include "SNode.h"
#include "seql_learn.h"

using namespace std;


SeqLearner::SeqLearner(){
	data_read = false;
}


bool SeqLearner::external_read (std::vector<std::string>& data) {
	// Set the max line/document size to (10Mb).
	// const int kMaxLineSize = 10000000;
	// char *line = new char[kMaxLineSize];
	char *column[5];
	unsigned int num_pos = 0;
	unsigned int num_neg = 0;
	int line_count = 0;
	string doc;

	cout << "\n\nread() input data....";

	for (int i = 0; i < data.size(); i++){

		line_count++;
		if (skip_items.find(line_count) != skip_items.end()) continue;


		char * line = new char[data[i].size() + 1];
		std::copy(data[i].begin(), data[i].end(), line);
		line[data[i].size()] = '\0'; // don't forget the terminating 0
		if (2 != tokenize (line, "\t ", column, 2)) {
			std::cerr << "FATAL: Format Error: " << line << std::endl;
			return false;
		}

		// Prepare class. _y is +1/-1.
		double _y = atof (column[0]);
		y.push_back (_y);

		if (_y > 0) num_pos++;
		else num_neg++;

		// Prepare doc. Assumes column[1] is the original text, e.g. no bracketing of original doc.
		doc.assign(column[1]);
		transaction.push_back(doc);

		//cout << "\nscanning docid: " << transaction.size() - 1 << " class y:" << _y << " doc:" << doc <<"*";
		cout.flush();
	}

	cout << "\n# positive samples: " << num_pos;
	cout << "\n# negative samples: " << num_neg;
	cout << "\nend read() input data....";

	data_read = true;
	return true;
}

bool SeqLearner::external_read (std::vector < std::string >& _transaction, std::vector < double >& _y){

	for (int i = 0; i < _transaction.size();i++){
		if (skip_items.find(i+1) != skip_items.end()) continue;
		transaction.push_back(_transaction[i]);
		y.push_back(_y[i]);
	}


	data_read = true;
	return true;
}


void SeqLearner::add_skips_items(int item){
	skip_items.insert(item);
}
// Read the input training documents, "true_class document" per line.
// A line in the training file can be: "+1 a b c"
bool SeqLearner::read (const char *filename) {
	//
	if (data_read){
		return true;
	}

	// Set the max line/document size to (10Mb).
	const int kMaxLineSize = 10000000;
	char *line = new char[kMaxLineSize];
	char *column[5];
	unsigned int num_pos = 0;
	unsigned int num_neg = 0;
	int line_count = 0;
	string doc;

	std::istream *ifs = 0;
	if (!strcmp(filename, "-")) {
		ifs = &std::cin;
	} else {
		ifs = new std::ifstream(filename);
		if (!*ifs) {
			std::cerr << "seql_learn" << " " << filename << " No such file or directory" << std::endl;
			return false;
		}
	}

	if (! ifs) return false;
	cout << "\n\nread() input data....";



	while (ifs->getline (line, kMaxLineSize)) {
		line_count++;
		if (skip_items.find(line_count) != skip_items.end()) continue;

		if (line[0] == '\0' || line[0] == ';') continue;
		if (line[strlen(line) - 1 ] == '\r')
			line[strlen(line) - 1 ] = '\0';
		//cout << "\nline size: " << strlen(line);
		//cout.flush();

		if (2 != tokenize (line, "\t ", column, 2)) {
			std::cerr << "FATAL: Format Error: " << line << std::endl;
			return false;
		}
		// Prepare class. _y is +1/-1.
		double _y = atof (column[0]);
		y.push_back (_y);

		if (_y > 0) num_pos++;
		else num_neg++;

		// Prepare doc. Assumes column[1] is the original text, e.g. no bracketing of original doc.
		doc.assign(column[1]);
		transaction.push_back(doc);

		//cout << "\nscanning docid: " << transaction.size() - 1 << " class y:" << _y << " doc:" << doc <<"*";
		cout.flush();
	}
	cout << "\n# positive samples: " << num_pos;
	cout << "\n# negative samples: " << num_neg;
	cout << "\nend read() input data....";

	delete [] line;

	data_read = true;
	return true;
}

// For current ngram, compute the gradient value and check prunning conditions.
// Update the current optimal ngram.
bool SeqLearner::can_prune_and_update_rule (rule_t& rule, SNode *space, unsigned int size) {

	++total;

	// Upper bound for the positive class.
	double upos = 0;
	// Upper bound for the negative class.
	double uneg = 0;
	// Gradient value at current ngram.
	double gradient = 0;
	// Support of current ngram.
	unsigned int support = 0;
	//string reversed_ngram;
	string ngram;
	std::vector <int>& loc = space->loc;
	//variable for y-beta^t*x (Squared loss)
	double y_btx=0;
	//variable for 1-y*beta^t*x (l2-svm)
	double ymbtx=0;
	// Compute the gradient and the upper bound on gradient of extensions.
	for (unsigned int i = 0; i < loc.size(); ++i) {
		if (loc[i] >= 0) continue;
		++support;
		unsigned int j = (unsigned int)(-loc[i]) - 1;

		// Choose objective function. 0: SLR, 2: l2-SVM 3: Squared loss.
		if (objective == 0) { //SLR
			// From differentiation we get a - in front of the sum_i_to_N
			gradient -= y[j] * exp_fraction[j];

			if (y[j] > 0) {
				upos -= y[j] * exp_fraction[j];
			} else {
				uneg -= y[j] * exp_fraction[j];
			}
		} //end SLR (logistic regression)

		else if (objective == 2) { //L2-SVM
			ymbtx = 1-y[j]*sum_best_beta[j];
			if (ymbtx > 0) {
				gradient -= 2 * y[j] * (ymbtx);

				if (y[j] > 0) {
					upos -= 2 * y[j] * (ymbtx);
				} else {
					uneg -= 2 * y[j] * (ymbtx);
				}
			}
		} //end l2-SVM

		else if (objective == 3) { //Squared loss
			y_btx=y[j]-sum_best_beta[j];
			gradient -= 2 * (y_btx);

			if (y_btx > 0) {
				upos -= 2 * (y_btx);
			} else {
				uneg -= 2 * (y_btx);
			}
		} //end Squared loss
	}

	// Assume we can prune until bound sais otherwise.
	bool flag_can_prune = 1;

	// In case C != 0 we need to update the gradient and check the exact bound
	// for non-zero features.
	if ( C != 0 ) {

		ngram = space->getNgram();

		if (verbosity > 3) {
			cout << "\ncurrent ngram rule: " << ngram;
			cout << "\nlocation size: " << space->loc.size() << "\n";
			cout << "\ngradient (before regularizer): " << gradient;
			cout << "\nupos (before regularizer): " << upos;
			cout << "\nuneg (before regularizer): " << uneg;
			cout << "\ntau: " << tau;
		}

		double current_upos = 0;
		double current_uneg = 0;

		// Retrieve the beta_ij coeficient of this feature. If beta_ij is non-zero,
		// update the gradient: gradient += C * [alpha*sign(beta_j) + (1-alpha)*beta_j];
		// Fct lower_bound return an iterator to the key >= given key.
		features_it = features_cache.lower_bound(ngram);
		// If there are keys starting with this prefix (this ngram starts at pos 0 in existing feature).
		if (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
			// If found an exact match for the key.
			// add regularizer to gradient.
			if (features_it->first.compare(ngram) == 0) {
				int sign = abs(features_it->second)/features_it->second;
				gradient += C * (alpha * sign + (1-alpha) * features_it->second);
			}

			if (verbosity > 3) {
				cout << "\ngradient after regularizer: " << gradient;
			}
			// Check if current feature s_j is a prefix of any non-zero features s_j'.
			// Check exact bound for every such non-zero feature.
			while (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
				int sign = abs(features_it->second)/features_it->second;
				current_upos = upos + C * (alpha * sign + (1-alpha) * features_it->second);
				current_uneg = uneg + C * (alpha * sign + (1-alpha) * features_it->second);

				if (verbosity > 3) {
					cout << "\nexisting feature starting with current ngram rule prefix: "
							<< features_it->first << ", " << features_it->second << ",  sign: " << sign;

					cout << "\ncurrent_upos: " << current_upos;
					cout << "\ncurrent_uneg: " << current_uneg;
					cout << "\ntau: " << tau;
				}
				// Check bound. If any non-zero feature starting with current ngram as a prefix
				// can still qualify for selection in the model,
				// we cannot prune the search space.
				if (std::max (abs(current_upos), abs(current_uneg)) > tau ) {
					flag_can_prune = 0;
					break;
				}
				++features_it;
			}
		} else {
			// If there are no features in the model starting with this prefix, check regular bound.
			if (std::max (abs(upos), abs(uneg)) > tau ) {
				flag_can_prune = 0;
			}
		}
	} // If C = 0, check regular bound.
	else {
		if (std::max (abs(upos), abs(uneg)) > tau ) {
			flag_can_prune = 0;
		}
	}

	if (support < minsup || flag_can_prune) {
		++pruned;
		if (verbosity > 3) {
			cout << "\n\nminsup || upper bound: pruned!\n";
		}
		return true;
	}

	double g = std::abs (gradient);
	// If current ngram better than previous best ngram, update optimal ngram.
	// Check min length requirement.
	if (g > tau && size >= minpat) {
		++rewritten;

		tau = g;
		rule.gradient = gradient;
		rule.size = size;

		if (C == 0) { // Retrieve the best ngram. If C != 0 this is already done.
			ngram = space->getNgram();
		} //end C==0.
		rule.ngram = ngram;

		if (verbosity >= 3) {
			cout << "\n\nnew current best ngram rule: " << rule.ngram;
			cout << "\ngradient: " << gradient << "\n";
		}

		rule.loc.clear ();
		for (unsigned int i = 0; i < space->loc.size(); ++i) {
			// Keep the doc ids where the best ngram appears.
			if (space->loc[i] < 0) rule.loc.push_back ((unsigned int)(-space->loc[i]) - 1);
		}
	}
	return false;
}

// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done breadth-first, e.g. grow all unigrams to bi-grams, than all bi-grams to tri-grams, etc.
void SeqLearner::span_bfs (rule_t& rule, SNode *space, std::vector<SNode *>& new_space, unsigned int size) {

	std::vector <SNode *>& next = space->next;

	// If working with gaps.
	// Check if number of consecutive gaps exceeds the max allowed.
	if(SNode::hasWildcardConstraints){
		if(space->violateWildcardConstraint()) return;
	}

	if (!(next.size() == 1 && next[0] == 0)) {
		// If there are candidates for extension, iterate through them, and try to prune some.
		if (! next.empty()) {
			if (verbosity > 4)
				cout << "\n !next.empty()";
			for (auto const &next_space : next) {
				// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for the prev ngram without the gap token.
				// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
				// so we can safely skip recomputing the gradient and bounds.
				if (next_space->ne.compare("*") == 0) {
					if (verbosity > 4)
						cout << "\nFeature ends in *, skipping gradient and bound computation.";
					new_space.push_back (next_space);
				} else if (! can_prune_and_update_rule (rule, next_space, size)) {
					new_space.push_back (next_space);
				}
			}
		} else {

			// Candidates obtained by extension.
			std::map<string, SNode> candidates;
			createCandidatesExpansions(space, candidates);
			// Keep only doc_ids for occurrences of current ngram.
			space->shrink ();
			if (candidates.size() == 0){
				next.push_back (0);
			}else{
				next.reserve(candidates.size());
				next.clear();
				// Prepare the candidate extensions.
				for (auto const &currCandidate : candidates) {

					auto c = new SNode;
					c->loc = currCandidate.second.loc;
					c->ne    = currCandidate.first;
					c->prev   = space;
					c->next.clear ();

					// Keep all the extensions of the current feature for future iterations.
					// If we need to save memory we could sacrifice this storage.
					next.push_back(c);

					// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
					// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
					// so we can safely skip recomputing the gradient and bounds.
					if (c->ne.compare("*") == 0) {
						if (verbosity > 3)
							cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next bfs level.";
						new_space.push_back (c);
					} else if (! can_prune_and_update_rule(rule, c, size)) {
						new_space.push_back (c);
					}
				}
			}
			// Adjust capacity of next vector
			std::vector<SNode *>(next).swap (next);
		} //end generation of candidates when they weren't stored already.
	}
}

void SeqLearner::createCandidatesExpansions(SNode* space, std::map<string, SNode>& candidates){
	// Prepare possible extensions.
	unsigned int docid = 0;

	std::vector <int>& loc = space->loc;

	// Iterate through the inverted index of the current feature.
	for (auto const& currLoc : loc) {
		// If current Location is negative it indicates a document rather than a possition in a document.
		if (currLoc < 0) {
			docid = (unsigned int)(-currLoc) - 1;
			continue;
		}

		// current unigram where extension is done
		string unigram = space->ne;
		if (verbosity > 4) {
			cout << "\ncurrent pos and start char: " <<  currLoc << " " << transaction[docid][currLoc];
			cout << "\ncurrent unigram to be extended (space->ne):" << unigram;
		}
		string next_unigram;
		// If not re-initialized, it should fail.
		unsigned int pos_start_next_unigram = transaction[docid].size();

		if (currLoc + unigram.size() < transaction[docid].size()) { //unigram is not in the end of doc, thus it can be extended.
			if (verbosity > 4) {
				cout << "\npotential extension...";
			}
			if (!SNode::tokenType) { // Word level token.

				// Find the first space after current unigram position.
				unsigned int pos_space = currLoc + unigram.size();
				// Skip over consecutive spaces.
				while ( (pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space]) ) {
					pos_space++;
				}
				// Check if doc ends in spaces, rather than a unigram.
				if (pos_space == transaction[docid].size()) {
					//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
					//std::exit(-1);
					continue;
				} else {
					pos_start_next_unigram = pos_space; //stoped at first non-space after consec spaces
					size_t pos_next_space = transaction[docid].find(' ', pos_start_next_unigram + 1);

					// Case1: the next unigram is in the end of the doc, no second space found.
					if (pos_next_space == string::npos) {
						next_unigram.assign(transaction[docid].substr(pos_start_next_unigram));
					} else { //Case2: the next unigram is inside the doc.
						next_unigram.assign(transaction[docid].substr(pos_start_next_unigram, pos_next_space - pos_start_next_unigram));
					}
				}
			} else { // Char level token. Skip spaces.
				if (!isspace(transaction[docid][currLoc + 1])) {
					//cout << "\nnext char is not space";
					next_unigram = transaction[docid][currLoc + 1]; //next unigram is next non-space char
					pos_start_next_unigram = currLoc + 1;
				} else { // If next char is space.
					unsigned int pos_space = currLoc + 1;
					// Skip consecutive spaces.
					while ((pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space])) {
						pos_space++;
					}
					// Check if doc ends in space, rather than a unigram.
					if (pos_space == transaction[docid].size()) {
						//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
						//std::exit(-1);
						continue;
					}
					/* //disallow using char-tokenization for space separated tokens.
	                           else {
	                           pos_start_next_unigram = pos_space;
	                           //cout << "\nnext next char is not space";
	                           next_unigram = transaction[docid][pos_start_next_unigram];
	                           } */
				}
			} //end char level token

			if (next_unigram.empty()) {
				//@THACH: just carry on if next unigram is empty
				if (verbosity > 4){
					cout <<"\nIn expansion for next_unigram: expansion of current unigram " << unigram << " is empty...continue";
				}
				continue;
				//cout <<"\nFATAL...in expansion for next_unigram: expansion of current unigram " << unigram << " is empty...exit";
				//std::exit(-1);
			}

			if (verbosity > 4) {
				cout << "\nnext_unigram for extension:" << next_unigram;
				cout << "\npos: " <<  pos_start_next_unigram << " " << transaction[docid][pos_start_next_unigram];
			}

			if (minsup == 1 || single_node_minsup_cache.find (next_unigram) != single_node_minsup_cache.end()) {
				candidates[next_unigram].add (docid, pos_start_next_unigram);
			}

			if (SNode::hasWildcardConstraints) {
				// If we allow gaps, we treat a gap as an additional unigram "*".
				// Its inverted index will simply be a copy pf the inverted index of the original features.
				// Example, for original feature "a", we extend to "a *", where inverted index of "a *" is simply
				// a copy of the inverted index of "a", except for positions where "a" is the last char in the doc.
				candidates["*"].add (docid, pos_start_next_unigram);
			}
		} //end generating candidates for the current pos

	} //end iteration through inverted index (docids iand pos) to find candidates
}


// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done deapth-first rather than breadth-first, e.g. grow each candidate to its longest unpruned sequence
void SeqLearner::span_dfs (rule_t& rule, SNode *space, unsigned int size) {

	std::vector <SNode *>& next = space->next;

	// Check if ngram larger than maxsize allowed.
	if (size > maxpat) return;

	if(SNode::hasWildcardConstraints){
		if(space->violateWildcardConstraint()) return;
	}

	if (!(next.size() == 1 && next[0] == 0)){
		// If the extensions are already computed, iterate through them and check pruning condition.
		if (! next.empty()) {
			if (verbosity >= 3)
				cout << "\n !next.empty()";
			for (std::vector<SNode*>::iterator it = next.begin(); it != next.end(); ++it) {
				if ((*it)->ne.compare("*") == 0) {
					if (verbosity > 3)
						cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
					// Expand each candidate DFS wise.
					span_dfs(rule, *it, size + 1);
				} else if (! can_prune_and_update_rule (rule, *it, size)) {
					// Expand each candidate DFS wise.
					span_dfs(rule, *it, size + 1);
				}
			}
		} else {

			// Candidates obtained by extension.
			std::map<string, SNode> candidates;
			createCandidatesExpansions(space, candidates);

			// Keep only doc_ids for occurrences of current ngram.
			space->shrink ();

			next.reserve(candidates.size());
			next.clear();
			// Prepare the candidate extensions.
			for (auto const &currCandidate : candidates) {

				SNode* c = new SNode;
				c->loc = currCandidate.second.loc;
				std::vector<int>(c->loc).swap(c->loc);
				c->ne    = currCandidate.first;
				c->prev   = space;
				c->next.clear ();

				// Keep all the extensions of the current feature for future iterations.
				// If we need to save memory we could sacrifice this storage.
				next.push_back (c);

				// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
				// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
				// so we can safely skip recomputing the gradient and bounds.
				if (c->ne.compare("*") == 0) {
					if (verbosity >= 3)
						cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
					span_dfs(rule, c, size + 1);
				} else // If this doesn't end in *, then check gradient and pruning condition.
					if (! can_prune_and_update_rule (rule, c, size)) {
						span_dfs(rule, c, size + 1);
					}
			}


			if (next.empty()) {
				next.push_back (0);
			}
			std::vector<SNode *>(next).swap (next);
		}
	}
}

// Line search method. Search for step size that minimizes loss.
// Compute loss in middle point of range, beta_n1, and
// for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
// Compare the loss for the 3 points, and choose range of 3 points
// which contains the minimum. Repeat until the range spanned by the 3 points is small enough,
// e.g. the range approximates well the vector where the loss function is minimized.
// Return the middle point of the best range.
void SeqLearner::find_best_range(vector<double>& sum_best_beta_n0, vector<double>& sum_best_beta_n1, vector<double>& sum_best_beta_n2,
		vector<double>& sum_best_beta_mid_n0_n1, vector<double>& sum_best_beta_mid_n1_n2,
		rule_t& rule, vector<double>* sum_best_beta_opt) {

	double min_range_size = 1e-3;
	double current_range_size = 0;
	int current_interpolation_iter = 0;

	long double loss_mid_n0_n1 = 0;
	long double loss_mid_n1_n2 = 0;
	long double loss_n1 = 0;

	for (unsigned int i = 0; i < transaction.size();  ++i) {
		if (verbosity > 4) {
			cout << "\nsum_best_beta_n0[i]: " << sum_best_beta_n0[i];
			cout << "\nsum_best_beta_n1[i]: " << sum_best_beta_n1[i];
			cout << "\nsum_best_beta_n2[i]: " << sum_best_beta_n2[i];
		}
		current_range_size += abs(sum_best_beta_n2[i] - sum_best_beta_n0[i]);
	}
	if (verbosity > 3)
		cout << "\ncurrent range size: " << current_range_size;

	double beta_coef_n1 = 0;
	double beta_coef_mid_n0_n1 = 0;
	double beta_coef_mid_n1_n2 = 0;

	if (C != 0 && sum_squared_betas != 0) {
		features_it = features_cache.find(rule.ngram);
	}
	// Start interpolation loop.
	while (current_range_size > min_range_size) {
		if (verbosity > 3)
			cout << "\ncurrent interpolation iteration: " << current_interpolation_iter;

		for (unsigned int i = 0; i < transaction.size();  ++i) { //loop through training samples
			sum_best_beta_mid_n0_n1[i] = (sum_best_beta_n0[i] + sum_best_beta_n1[i]) / 2;
			sum_best_beta_mid_n1_n2[i] = (sum_best_beta_n1[i] + sum_best_beta_n2[i]) / 2;

			if ( C != 0) {
				beta_coef_n1 = sum_best_beta_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
				beta_coef_mid_n0_n1 = sum_best_beta_mid_n0_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
				beta_coef_mid_n1_n2 = sum_best_beta_mid_n1_n2[rule.loc[0]] - sum_best_beta[rule.loc[0]];
			}

			if (verbosity > 4) {
				cout << "\nsum_best_beta_mid_n0_n1[i]: " << sum_best_beta_mid_n0_n1[i];
				cout << "\nsum_best_beta_mid_n1_n2[i]: " << sum_best_beta_mid_n1_n2[i];
			}
			loss_n1 += computeLossTerm(sum_best_beta_n1[i], y[i]);
			loss_mid_n0_n1 += computeLossTerm(sum_best_beta_mid_n0_n1[i], y[i]);
			loss_mid_n1_n2 += computeLossTerm(sum_best_beta_mid_n1_n2[i], y[i]);
		} //end loop through training samples.

		if ( C != 0 ) {
			// Add the Elastic Net regularization term.
			// If this is the first ngram selected.
			if (sum_squared_betas == 0) {
				loss_n1 = loss_n1 + C * (alpha * abs(beta_coef_n1) + (1-alpha) * 0.5 * pow(beta_coef_n1, 2));
				loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * abs(beta_coef_mid_n0_n1) + (1-alpha) * 0.5 * pow(beta_coef_mid_n0_n1, 2));
				loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * abs(beta_coef_mid_n1_n2) + (1-alpha) * 0.5 * pow(beta_coef_mid_n1_n2, 2));
			} else {
				// If this feature was not selected before.
				if (features_it == features_cache.end()) {
					loss_n1 = loss_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_n1, 2)));
					loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n0_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n0_n1, 2)));
					loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n1_n2)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n1_n2, 2)));
				} else {
					double new_beta_coef_n1 = features_it->second + beta_coef_n1;
					double new_beta_coef_mid_n0_n1 = features_it->second + beta_coef_mid_n0_n1;
					double new_beta_coef_mid_n1_n2 = features_it->second + beta_coef_mid_n1_n2;
					loss_n1 = loss_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_n1, 2)));
					loss_mid_n0_n1 = loss_mid_n0_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n0_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n0_n1, 2)));
					loss_mid_n1_n2 = loss_mid_n1_n2  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n1_n2)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n1_n2, 2)));
				}
			}
		}// end check C != 0.

		// Focus on the range that contains the minimum of the loss function.
		// Compare the 3 points beta_n, and mid_beta_n-1_n and mid_beta_n_n+1.
		if (loss_n1 <= loss_mid_n0_n1 && loss_n1 <= loss_mid_n1_n2) {
			// Min is in beta_n1.
			if (verbosity > 4) {
				cout << "\nmin is sum_best_beta_n1";
				cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
				cout << "\nloss_n1: " << loss_n1;
				cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
			}
			// Make the beta_n0 be the beta_mid_n0_n1.
			sum_best_beta_n0.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
			// Make the beta_n2 be the beta_mid_n1_n2.
			sum_best_beta_n2.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
		}
		else {
			if (loss_mid_n0_n1 <= loss_n1 && loss_mid_n0_n1 <= loss_mid_n1_n2) {
				// Min is beta_mid_n0_n1.
				if (verbosity > 4) {
					cout << "\nmin is sum_best_beta_mid_n0_n1";
					cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
					cout << "\nloss_n1: " << loss_n1;
					cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
				}
				// Make the beta_n2 be the beta_n1.
				sum_best_beta_n2.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
				// Make the beta_n1 be the beta_mid_n0_n1.
				sum_best_beta_n1.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
			} else {
				// Min is beta_mid_n1_n2.
				if (verbosity > 4) {
					cout << "\nmin is sum_best_beta_mid_n1_n2";
					cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
					cout << "\nloss_n1: " << loss_n1;
					cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
				}
				// Make the beta_n0 be the beta_n1.
				sum_best_beta_n0.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
				// Make the beta_n1 be the beta_mid_n1_n2
				sum_best_beta_n1.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
			}
		}

		++current_interpolation_iter;
		loss_mid_n0_n1 = 0;
		loss_mid_n1_n2 = 0;
		loss_n1 = 0;
		current_range_size = 0;

		for (unsigned int i = 0; i < transaction.size();  ++i) {
			if (verbosity > 4) {
				cout << "\nsum_best_beta_n0[i]: " << sum_best_beta_n0[i];
				cout << "\nsum_best_beta_n1[i]: " << sum_best_beta_n1[i];
				cout << "\nsum_best_beta_n2[i]: " << sum_best_beta_n2[i];
			}
			current_range_size += abs(sum_best_beta_n2[i] - sum_best_beta_n0[i]);
		}
		if (verbosity > 4) {
			cout << "\ncurrent range size: " << current_range_size;
		}
	} // end while loop.

	// Keep the middle point of the best range.
	for (unsigned int i = 0; i < transaction.size();  ++i) {
		sum_best_beta_opt->push_back(sum_best_beta_n1[i]);
		// Trust this step only by a fraction.
		//sum_best_beta_opt->push_back(0.5 * sum_best_beta[i] + 0.5 * sum_best_beta_n1[i]);
	}
} // end find_best_range().

// Line search method. Binary search for optimal step size. Calls find_best_range(...).
// sum_best_beta keeps track of the scalar product beta_best^t*xi for each doc xi.
// Instead of working with the new weight vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n)
// we work directly with the scalar product.
// We output the sum_best_beta_opt which contains the scalar poduct of the optimal beta found, by searching for the optimal
// epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
// epsilon is the starting value
// rule contains info about the gradient at the current iteration
void SeqLearner::binary_line_search(rule_t& rule, vector<double>* sum_best_beta_opt) {

	sum_best_beta_opt->clear();
	// Starting value for parameter in step size search.
	// Set the initial epsilon value small enough to guaranteee
	// log-like increases in the first steps.
	double exponent = ceil(log10(abs(rule.gradient)));
	double epsilon = min(1e-3, pow(10, -exponent));

	if (verbosity > 3) {
		cout << "\nrule.ngram: " << rule.ngram;
		cout << "\nrule.gradient: " << rule.gradient;
		cout << "\nexponent of epsilon: " << -exponent;
		cout << "\nepsilon: " << epsilon;
	}

	// Keep track of scalar product at points beta_n-1, beta_n and beta_n+1.
	// They are denoted with beta_n0, beta_n1, beta_n2.
	//vector<double> sum_best_beta_n0(sum_best_beta.size());
	vector<double> sum_best_beta_n0(sum_best_beta);
	vector<double> sum_best_beta_n1(sum_best_beta);
	vector<double> sum_best_beta_n2(sum_best_beta);

	// Keep track of loss at the three points, n0, n1, n2.
	long double loss_n0 = 0;
	long double loss_n1 = 0;
	long double loss_n2 = 0;

	// Binary search for epsilon. Similar to bracketing phase in which
	// we search for some range with promising epsilon.
	// The second stage finds the epsilon or corresponding weight vector with smallest l2-loss value.

	// **************************************************************************/
	// As long as the l2-loss decreases, double the epsilon.
	// Keep track of the last three values of beta, or correspondingly
	// the last 3 values for the scalar product of beta and xi.
	int n = 0;

	if ( C != 0 && sum_squared_betas != 0) {
		features_it = features_cache.find(rule.ngram);
	}

	double beta_coeficient_update = 0;
	do {
		if (verbosity > 3)
			cout << "\nn: " << n;

		// For each location (e.g. docid), update the score of the documents containing best rule.
		// E.g. update beta^t * xi.
		beta_coeficient_update -= pow(2, n * 1.0) * epsilon * rule.gradient;
		for (unsigned int i = 0; i < rule.loc.size(); ++i) {
			sum_best_beta_n0[rule.loc[i]] = sum_best_beta_n1[rule.loc[i]];
			sum_best_beta_n1[rule.loc[i]] = sum_best_beta_n2[rule.loc[i]];
			sum_best_beta_n2[rule.loc[i]] = sum_best_beta_n1[rule.loc[i]] - pow(2, n * 1.0) * epsilon * rule.gradient;

			if (verbosity > 4 && i == 0) {
				cout << "\nsum_best_beta_n0[rule.loc[i]: " << sum_best_beta_n0[rule.loc[i]];
				cout << "\nsum_best_beta_n1[rule.loc[i]: " << sum_best_beta_n1[rule.loc[i]];
				cout << "\nsum_best_beta_n2[rule.loc[i]: " << sum_best_beta_n2[rule.loc[i]];
			}
		}

		// Compute loss for all 3 values: n-1, n, n+1
		// In the first iteration compute necessary loss.
		if (n == 0) {
			loss_n0 = loss;
			loss_n1 = loss;
		} else {
			// Update just loss_n2.
			// The loss_n0 and loss_n1 are already computed.
			loss_n0 = loss_n1;
			loss_n1 = loss_n2;
		}
		loss_n2 = 0;
		computeLoss(loss_n2, sum_best_beta_n2);

		if (verbosity > 4) {
			cout << "\nloss_n2 before adding regularizer: " << loss_n2;
		}
		// If C != 0, add the L2 regularizer term to the l2-loss.
		// If this is the first ngram selected.
		if ( C != 0 ) {
			if (sum_squared_betas == 0) {
				loss_n2 = loss_n2 + C * (alpha * abs(beta_coeficient_update) + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));

				if (verbosity > 4) {
					cout << "\nregularizer: " << C * (alpha * abs(beta_coeficient_update) + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));
				}
			} else {
				// If this feature was not selected before.
				if (features_it == features_cache.end()) {
					loss_n2 = loss_n2 + C * (alpha * (sum_abs_betas + abs(beta_coeficient_update)) + (1 - alpha) * 0.5 * (sum_squared_betas +
							pow(beta_coeficient_update, 2)));
				} else {
					double new_beta_coeficient = features_it->second + beta_coeficient_update;
					loss_n2 = loss_n2 + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coeficient)) + (1 - alpha) * 0.5 *                                    (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coeficient, 2)));
				}
			}
		} // end C != 0.
		if (verbosity > 4) {
			cout << "\nloss_n0: " << loss_n0;
			cout << "\nloss_n1: " << loss_n1;
			cout << "\nloss_n2: " << loss_n2;
		}
		++n;
	} while (loss_n2 < loss_n1);
	// **************************************************************************/

	if (verbosity > 3)
		cout << "\nFinished doubling epsilon! The monotonicity loss_n+1 < loss_n is broken!";

	// Search for the beta in the range beta_n-1, beta_mid_n-1_n, beta_n, beta_mid_n_n+1, beta_n+1
	// that minimizes the objective function. It suffices to compare the 3 points beta_mid_n-1_n, beta_n, beta_mid_n_n+1,
	// as the min cannot be achieved at the extrem points of the range.
	// Take the 3 point range containing the point that achieves minimum loss.
	// Repeat until the 3 point range is too small, or a fixed number of iterations is achieved.

	// **************************************************************************/
	vector<double> sum_best_beta_mid_n0_n1(sum_best_beta.size());
	vector<double> sum_best_beta_mid_n1_n2(sum_best_beta.size());

	find_best_range(sum_best_beta_n0, sum_best_beta_n1, sum_best_beta_n2,
			sum_best_beta_mid_n0_n1, sum_best_beta_mid_n1_n2,
			rule, sum_best_beta_opt);
	// **************************************************************************/
} // end binary_line)search().

// Searches the space of all subsequences for the ngram with the ngram with the maximal abolute gradient and saves it in rule
SeqLearner::rule_t SeqLearner::findBestNgram(rule_t& rule ,std::vector <SNode*>& old_space, std::vector<SNode*>& new_space, std::map<string, SNode>& seed){

	// Reset
	tau = 0;
	rule.ngram = "";
	rule.gradient = 0;
	rule.loc.clear ();
	rule.size = 0;
	old_space.clear ();
	new_space.clear ();
	pruned = total = rewritten = 0;

	// Iterate through unigrams.
	for (auto &unigram : seed) {
		if (!can_prune_and_update_rule (rule, &unigram.second, 1)) {
			// Check BFS vs DFS traversal.
			if (!traversal_strategy) {
				old_space.push_back (&unigram.second);
			} else {
				// Traversal is DFS.
				span_dfs (rule, &unigram.second, 2);
			}
		}
	}

	// If BFS traversal.
	if (!traversal_strategy) {
		// Search for best n-gram. Try to extend in a bfs fashion,
		// level per level, e.g., first extend unigrams to bigrams, then bigrams to trigrams, etc.
		//*****************************************************/
		for (unsigned int size = 2; size <= maxpat; ++size) {
			for (unsigned int i = 0; i < old_space.size(); ++i) {
				span_bfs (rule, old_space[i], new_space, size);
			}
			if (new_space.empty()) {
				break;
			}
			old_space = new_space;
			new_space.clear ();
		} // End search for best n-gram.
	} // end check BFS traversal.

	if (verbosity >= 2) {
		cout << "\nfound best ngram! ";
		cout << "\nrule.gradient: " << rule.gradient;
		// gettimeofday(&t, NULL);
		// cout << " (per iter: " << t.tv_sec - t_start_iter.tv_sec << " seconds; " << (t.tv_sec - t_start_iter.tv_sec) / 60.0 << " minutes; total time:"
		// 		<< (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes)";
	}
	return rule;
};


int SeqLearner::run (const char *in,
		const char *out,
		unsigned int _objective,
		unsigned int _maxpat,
		unsigned int _minpat,
		unsigned int maxitr,
		unsigned int _minsup,
		unsigned int _maxgap,
		unsigned int _maxcongap,
		bool _token_type,
		bool _traversal_strategy,
		double _convergence_threshold,
		double _regularizer_value,
		double _l1vsl2_regularizer,
		int _verbosity) {

	objective = _objective;
	maxpat = _maxpat;
	minpat = _minpat;
	minsup = _minsup;
	SNode::setupWildcardConstraint(_maxgap, _maxcongap);
	SNode::tokenType = _token_type;
	traversal_strategy = _traversal_strategy;
	convergence_threshold = _convergence_threshold;
	C = _regularizer_value;
	alpha = _l1vsl2_regularizer;
	verbosity = _verbosity;

	// gettimeofday(&t_origin, NULL);

	// TODO setup ofstream in properway
	std::ofstream os (out);
	if(!setup(in, out, os)){
		return -1;
	};
	// A map from unigrams to search_space.
	std::map <string, SNode> seed;

	prepareInvertedIndex(seed);
	deleteUndersupportedUnigrams(seed);

	std::vector <SNode*> old_space;
	std::vector <SNode*> new_space;

	// The optimal step length.
	double step_length_opt;
	// Set the convergence threshold as in paper by Madigan et al on BBR.
	//double convergence_threshold = 0.005;
	double convergence_rate;

	// Current rule.
	rule_t       rule;
	double sum_abs_scalar_prod_diff;
	double sum_abs_scalar_prod;
	loss = 0.0;

	// Compute loss with start beta vector.
	computeLoss(loss, sum_best_beta);

	if (verbosity >= 1) {
		cout << "\nstart loss: " << loss;
	}

	// Loop for number of given optimization iterations.
	for (unsigned int itr = 0; itr < maxitr; ++itr) {
		// gettimeofday(&t_start_iter, NULL);

		// Search in the feature space for the Ngram with the best absolute gradient value
		findBestNgram(rule, old_space, new_space, seed);

		// rule contains the best best ngram

		// Checck if we found ngram with non zero gradient
		if(rule.loc.size() == 0){
			cout<<"\nBest ngram has a gradient of 0 => Stop search\n";
			break;
		}
		// Use line search to detect the best step_length/learning_rate.
		// The method does binary search for the step length, by using a start parameter epsilon.
		// It works by first bracketing a promising value, followed by focusing on the smallest interval containing this value.
		binary_line_search(rule, &sum_best_beta_opt);

		// The optimal step length as obtained from the line search.
		step_length_opt = sum_best_beta_opt[rule.loc[0]] - sum_best_beta[rule.loc[0]];
		//cout << "\nOptimal step length: " << step_length_opt;

		// Update the weight of the best n-gram.
		// Insert or update new feature.
		if ( C != 0 ) {
			map<string, double>::iterator features_it = features_cache.find(rule.ngram);
			if (features_it == features_cache.end()) {
				// If feature not there, insert it.
				features_cache[rule.ngram] = step_length_opt;
			} else {// Adjust coeficient and the sums of coeficients.
				sum_squared_betas = sum_squared_betas - pow(features_it->second, 2);
				sum_abs_betas = sum_abs_betas - abs(features_it->second);
				features_it->second += step_length_opt;
			}
			sum_squared_betas += pow(features_cache[rule.ngram], 2);
			sum_abs_betas += abs(features_cache[rule.ngram]);

		}
		sum_abs_scalar_prod_diff = 0;
		sum_abs_scalar_prod = 0;
		// Remember the loss from prev iteration.
		old_loss = loss;
		loss = 0;
		computeLoss(loss, sum_best_beta_opt, sum_abs_scalar_prod_diff, sum_abs_scalar_prod, exp_fraction);

		if (verbosity >= 2) {
			cout << "\nloss: " << loss;
			if ( C != 0 ) {
				cout << "\npenalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
			}
		}
		// Update the log-likelihood with the regularizer term.
		if ( C != 0 ) {
			loss = loss + C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}

		//stop if loss doesn't improve; a failsafe check on top of conv_rate (based on predicted score residuals) reaching conv_threshold
		if (old_loss - loss == 0) {
			if (verbosity >= 1) {
				cout << "\n\nFinish iterations due to: no change in loss value!";
				cout << "\nloss + penalty term: " << loss;
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		}

		// The optimal step length as obtained from the line search.
		// Stop the alg if weight of best grad feature is below a given threshold.
		// Inspired by paper of Liblinear people that use a thereshold on the value of the gradient to stop close to optimal solution.
		if (abs(step_length_opt) > 1e-8)
			os << step_length_opt << ' ' << rule.ngram << "\n";
		else {
			if (verbosity >= 1) {
				cout << "\n\nFinish iterations due to: step_length_opt <= 1e-8 (due to numerical precision loss doesn't improve for such small weights)!";
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		}

		if (verbosity >= 2) {
			std::cout <<  "\n #itr: " << itr << " #features: " << features_cache.size () << " #rewrote: " << rewritten << " #prone: " << pruned << " #total: " << total << " stepLength: "
					<< step_length_opt << " rule: " << rule.ngram;

			cout << "\nloss + penalty term: " << loss;
			cout.flush();
		}

		// Set the convergence rate as in paper by Madigan et al on BBR.
		convergence_rate = sum_abs_scalar_prod_diff / (1 + sum_abs_scalar_prod);

		if (convergence_rate <= convergence_threshold) {
			if (verbosity >= 1) {
				cout << "\nconvergence rate: " << convergence_rate;
				cout << "\n\nFinish iterations due to: convergence test (convergence_thereshold=" << convergence_threshold << ")!";
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		} // Otherwise, loop up to the user provided # iter or convergence threshold.

		//sum_best_beta is the optimum found using line search
		sum_best_beta.assign(sum_best_beta_opt.begin(), sum_best_beta_opt.end());

	} //end optimization iterations.

	// gettimeofday(&t, NULL);
	if (verbosity >= 1) {
		if ( C != 0 ) {
			cout << "\nend penalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}
		cout << "\nend loss + penalty_term: " << loss;
		// cout << "\n\ntotal time: " << t.tv_sec - t_origin.tv_sec << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes\n ";
	}
	return 1;
} //end run().

int SeqLearner::run_internal (
		std::map<std::string,double> &model,
		unsigned int _objective,
		unsigned int _maxpat,
		unsigned int _minpat,
		unsigned int maxitr,
		unsigned int _minsup,
		unsigned int _maxgap,
		unsigned int _maxcongap,
		bool _token_type,
		bool _traversal_strategy,
		double _convergence_threshold,
		double _regularizer_value,
		double _l1vsl2_regularizer,
		int _verbosity){
	objective = _objective;
	maxpat = _maxpat;
	minpat = _minpat;
	minsup = _minsup;
	SNode::setupWildcardConstraint(_maxgap, _maxcongap);
	SNode::tokenType = _token_type;
	traversal_strategy = _traversal_strategy;
	convergence_threshold = _convergence_threshold;
	C = _regularizer_value;
	alpha = _l1vsl2_regularizer;
	verbosity = _verbosity;

	// gettimeofday(&t_origin, NULL);

	setup_internal();

	// A map from unigrams to search_space.
	std::map <string, SNode> seed;

	prepareInvertedIndex(seed);
	deleteUndersupportedUnigrams(seed);

	std::vector <SNode*> old_space;
	std::vector <SNode*> new_space;

	// The optimal step length.
	double step_length_opt;
	// Set the convergence threshold as in paper by Madigan et al on BBR.
	//double convergence_threshold = 0.005;
	double convergence_rate;

	// Current rule.
	rule_t       rule;
	double sum_abs_scalar_prod_diff;
	double sum_abs_scalar_prod;
	loss = 0.0;
	// sum of steps
	double alpha_sum = 0.0;
	// Compute loss with start beta vector.
	computeLoss(loss, sum_best_beta);

	if (verbosity >= 1) {
		cout << "\nstart loss: " << loss;
	}

	// Loop for number of given optimization iterations.
	for (unsigned int itr = 0; itr < maxitr; ++itr) {
		// gettimeofday(&t_start_iter, NULL);

		// Search in the feature space for the Ngram with the best absolute gradient value
		findBestNgram(rule, old_space, new_space, seed);

		// rule contains the best best ngram

		// Checck if we found ngram with non zero gradient
		if(rule.loc.size() == 0){
			cout<<"\nBest ngram has a gradient of 0 => Stop search\n";
			break;
		}
		// Use line search to detect the best step_length/learning_rate.
		// The method does binary search for the step length, by using a start parameter epsilon.
		// It works by first bracketing a promising value, followed by focusing on the smallest interval containing this value.
		binary_line_search(rule, &sum_best_beta_opt);

		// The optimal step length as obtained from the line search.
		step_length_opt = sum_best_beta_opt[rule.loc[0]] - sum_best_beta[rule.loc[0]];
		//cout << "\nOptimal step length: " << step_length_opt;

		// Update the weight of the best n-gram.
		// Insert or update new feature.
		if ( C != 0 ) {
			map<string, double>::iterator features_it = features_cache.find(rule.ngram);
			if (features_it == features_cache.end()) {
				// If feature not there, insert it.
				features_cache[rule.ngram] = step_length_opt;
			} else {// Adjust coeficient and the sums of coeficients.
				sum_squared_betas = sum_squared_betas - pow(features_it->second, 2);
				sum_abs_betas = sum_abs_betas - abs(features_it->second);
				features_it->second += step_length_opt;
			}
			sum_squared_betas += pow(features_cache[rule.ngram], 2);
			sum_abs_betas += abs(features_cache[rule.ngram]);

		}
		sum_abs_scalar_prod_diff = 0;
		sum_abs_scalar_prod = 0;
		// Remember the loss from prev iteration.
		old_loss = loss;
		loss = 0;
		computeLoss(loss, sum_best_beta_opt, sum_abs_scalar_prod_diff, sum_abs_scalar_prod, exp_fraction);

		if (verbosity >= 2) {
			cout << "\nloss: " << loss;
			if ( C != 0 ) {
				cout << "\npenalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
			}
		}
		// Update the log-likelihood with the regularizer term.
		if ( C != 0 ) {
			loss = loss + C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}

		//stop if loss doesn't improve; a failsafe check on top of conv_rate (based on predicted score residuals) reaching conv_threshold
		if (old_loss - loss == 0) {
			if (verbosity >= 1) {
				cout << "\n\nFinish iterations due to: no change in loss value!";
				cout << "\nloss + penalty term: " << loss;
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		}

		// The optimal step length as obtained from the line search.
		// Stop the alg if weight of best grad feature is below a given threshold.
		// Inspired by paper of Liblinear people that use a thereshold on the value of the gradient to stop close to optimal solution.
		if (abs(step_length_opt) > 1e-8){

			//cout << step_length_opt << ' ' << rule.ngram << "\n";
			//cout << model[rule.ngram] << endl;
			alpha_sum += abs(step_length_opt);
			model[rule.ngram] += step_length_opt;
			//if (model.find(rule.ngram) != model.end()){
			//	model[rule.ngram] = step_length_opt;
			//} else {
			//	model[rule.ngram] += step_length_opt;
			//}
		}
		else {
			if (verbosity >= 1) {
				cout << "\n\nFinish iterations due to: step_length_opt <= 1e-8 (due to numerical precision loss doesn't improve for such small weights)!";
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		}

		if (verbosity >= 2) {
			std::cout <<  "\n #itr: " << itr << " #features: " << features_cache.size () << " #rewrote: " << rewritten << " #prone: " << pruned << " #total: " << total << " stepLength: "
					<< step_length_opt << " rule: " << rule.ngram;

			cout << "\nloss + penalty term: " << loss;
			cout.flush();
		}

		// Set the convergence rate as in paper by Madigan et al on BBR.
		convergence_rate = sum_abs_scalar_prod_diff / (1 + sum_abs_scalar_prod);

		if (convergence_rate <= convergence_threshold) {
			if (verbosity >= 1) {
				cout << "\nconvergence rate: " << convergence_rate;
				cout << "\n\nFinish iterations due to: convergence test (convergence_thereshold=" << convergence_threshold << ")!";
				cout << "\n# iterations: " << itr + 1;
			}
			break;
		} // Otherwise, loop up to the user provided # iter or convergence threshold.

		//sum_best_beta is the optimum found using line search
		sum_best_beta.assign(sum_best_beta_opt.begin(), sum_best_beta_opt.end());

	} //end optimization iterations.

	// gettimeofday(&t, NULL);
	if (verbosity >= 1) {
		if ( C != 0 ) {
			cout << "\nend penalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}
		cout << "\nend loss + penalty_term: " << loss;
		// cout << "\n\ntotal time: " << t.tv_sec - t_origin.tv_sec << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes\n ";
	}


	//for (std::map<std::string, double>::iterator it = model.begin(); it != model.end(); ++it) {
	//	alpha_sum += abs(it->second);
	//}
	for (std::map<std::string, double>::iterator it = model.begin(); it != model.end(); ++it) {
			it->second = it->second * 2 / alpha_sum;
	}

	deleteTree(seed);

	return 1;
}

void SeqLearner::deleteTree(std::map<std::string, SNode>& seed){
	// maybe it's not necessary ?

}


void SeqLearner::prepareInvertedIndex (std::map<string, SNode>& seed) {
	string unigram;
	bool at_space = false;

	// Prepare the locations for unigrams.
	if (verbosity >= 1) {
		cout << "\nprepare inverted index for unigrams";
	}

	for (unsigned int docid = 0; docid < transaction.size(); ++docid) {
		at_space = false;
		//cout << "\nscanning docid: " << docid << ", class y: " << y[docid] << "\n";
		for (unsigned int pos = 0; pos < transaction[docid].size(); ++pos) {
			// Skip white spaces. They are not considered as unigrams.
			if (isspace(transaction[docid][pos])) {
				at_space = true;
				continue;
			}
			// If word level tokens.
			if (!SNode::tokenType) {
				if (at_space) {
					at_space = false;
					if (!unigram.empty()) {
						SNode & tmp = seed[unigram];
						tmp.add (docid,pos - unigram.size() - 1);
						tmp.next.clear ();
						tmp.ne = unigram;
						tmp.prev = 0;
						unigram.clear();
					}
					unigram.push_back(transaction[docid][pos]);
				} else {
					unigram.push_back(transaction[docid][pos]);
				}
			} else {
				//@THACH: allow space for boss-learning
				//if (at_space) {
				//	//Previous char was a space.
				//	//Disallow using char-tokenization for space-separated tokens. It confuses the features, whether to add " " or not.
				//			cout << "\nFATAL...found space in docid: " << docid << ", at position: " << pos-1;
				//			cout << "\nFATAL...char-tokenization assumes contiguous tokens (i.e., tokens are not separated by space).";
				//			cout << "\nFor space separated tokens please use word-tokenization or remove spaces to get valid input for char-tokenization.";
				//			cout << "\n...Exiting.....\n";
				//			std::exit(-1);
				//}
				// Char (i.e. byte) level token.
				unigram = transaction[docid][pos];
				SNode & tmp = seed[unigram];
				tmp.add (docid,pos);
				tmp.next.clear ();
				tmp.ne = unigram;
				tmp.prev = 0;
				unigram.clear();
			}
		} //end for transaction.
		// For word-tokens take care of last word of doc.
		if (!SNode::tokenType) {
			if (!unigram.empty()) {
				SNode & tmp = seed[unigram];
				tmp.add (docid, transaction[docid].size() - unigram.size());
				tmp.next.clear ();
				tmp.ne = unigram;
				tmp.prev = 0;
				unigram.clear();
			}
		}
	} //end for docid.

	// gettimeofday(&t, NULL);
	// if (verbosity >= 1) {
		// cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
		// cout.flush();
	// }
};

void SeqLearner::deleteUndersupportedUnigrams(std::map<string, SNode>& seed){
	// Keep only unigrams above minsup threshold.
	for (auto it= seed.cbegin(); it != seed.cend();) {
		if (it->second.support() < minsup) {
			if (verbosity >= 1) {
				cout << "\nremove unigram (minsup):" << it->first;
				cout.flush();
			}
			seed.erase(it++);
		} else {
			single_node_minsup_cache.insert (it->second.ne);
			if (verbosity >= 1) {
				cout << "\ndistinct unigram:" << it->first;
			}
			++it;
		}
	}
	if( single_node_minsup_cache.size()==0){
		cout << "\n>>> NO UNIGRAM LEFT\nMaybe adjust the minsup parameter";
		exit(1);
	};
	// gettimeofday(&t, NULL);
	if (verbosity >= 1) {
		cout << "\n# distinct unigrams: " << single_node_minsup_cache.size();
		// cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
		cout.flush();
	}
};


long double SeqLearner::computeLossTerm(const double& beta, const double &y){
	if (objective == 0) { //SLR
		if (-y * beta > 8000) {
			return log(LDBL_MAX);
		} else {
			return log(1 + exp(-y * beta));
		}
	} //end SLR

	else if (objective == 2) { //L2-SVM
		if (1 - y * beta > 0)
			return pow(1 - y * beta, 2);
	} //end L2-SVM

	else if (objective == 3 ) { //Squared loss
		return pow(y - beta, 2);
	} //end Squared loss
	return 0;
}
long double SeqLearner::computeLossTerm(const double& beta, const double &y, long double &exp_fraction){
	if (objective == 0) { //SLR
		if (y * beta > 8000) {
			exp_fraction = 0;
		} else {
			exp_fraction = 1 / (1 + exp(y * beta));
		}

		if (-y * beta > 8000) {
			return log(LDBL_MAX);
		} else {
			return log(1 + exp(-y * beta));
		}
	} //end SLR

	else if (objective == 2) { //L2-SVM
		if (1 - y * beta > 0)
			return pow(1 - y * beta, 2);
	} //end L2-SVM

	else if (objective == 3 ) { //Squared loss
		return pow(y - beta, 2);
	}//end Squared loss
	return 0;
}
void SeqLearner::computeLoss(long double &loss, const std::vector<double>& beta){
	for (unsigned int i = 0; i < transaction.size();  ++i) {
		loss += computeLossTerm(beta[i],y[i]);
	}
}

void SeqLearner::computeLoss(long double &loss, const std::vector<double>& beta,
		double &sum_abs_scalar_prod_diff, double &sum_abs_scalar_prod , std::vector<double long>& exp_fraction){
	for (unsigned int i = 0; i < transaction.size();  ++i) {
		loss += computeLossTerm(beta[i],y[i], exp_fraction[i]);
		// Compute the sum of per document difference between the scalar product at 2 consecutive iterations.
		sum_abs_scalar_prod_diff += abs(beta[i] - sum_best_beta[i]);
		// Compute the sum of per document scalar product at current iteration.
		sum_abs_scalar_prod += abs(beta[i]);
	}
}
void SeqLearner::computeLoss(long double &loss, const std::vector<double>& beta,
		double &sum_abs_scalar_prod_diff, double &sum_abs_scalar_prod ){
	for (unsigned int i = 0; i < transaction.size();  ++i) {
		loss += computeLossTerm(beta[i],y[i]);
		// Compute the sum of per document difference between the scalar product at 2 consecutive iterations.
		sum_abs_scalar_prod_diff += abs(beta[i] - sum_best_beta[i]);
		// Compute the sum of per document scalar product at current iteration.
		sum_abs_scalar_prod += abs(beta[i]);
	}
}

bool SeqLearner::setup(const char *in, const char *out, std::ofstream& os){
	if ( !read (in)) {
		std::cerr << "FATAL: Cannot open input file: " << in << std::endl;
		return false;
	}
	// gettimeofday(&t, NULL);
	// if (verbosity > 0){
		// cout << "( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
	//	cout.flush();
	// }

	if (! os) {
		std::cerr << "FATAL: Cannot open output file: " << out << std::endl;
		return false;
	}

	std::cout.setf(std::ios::fixed,std::ios::floatfield);
	std::cout.precision(8);

	os.setf(std::ios::fixed,std::ios::floatfield);
	os.precision(12);

	unsigned int l   = transaction.size();

	// All beta coeficients are zero when starting.
	sum_squared_betas = 0;
	sum_abs_betas = 0;

	sum_best_beta.resize(l);
	// The starting point is beta = (0, 0, 0, 0, .....).
	std::fill(sum_best_beta.begin(), sum_best_beta.end(), 0.0);
	exp_fraction.resize (l);
	std::fill (exp_fraction.begin(), exp_fraction.end(), 1.0 /2.0);

	return true;
}

bool SeqLearner::setup_internal(){

	// gettimeofday(&t, NULL);
	// if (verbosity > 0){
	// 	cout << "( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
	// 	cout.flush();
	// }



	std::cout.setf(std::ios::fixed,std::ios::floatfield);
	std::cout.precision(8);


	unsigned int l   = transaction.size();

	// All beta coeficients are zero when starting.
	sum_squared_betas = 0;
	sum_abs_betas = 0;

	sum_best_beta.resize(l);
	// The starting point is beta = (0, 0, 0, 0, .....).
	std::fill(sum_best_beta.begin(), sum_best_beta.end(), 0.0);
	exp_fraction.resize (l);
	std::fill (exp_fraction.begin(), exp_fraction.end(), 1.0 /2.0);

	return true;
}
