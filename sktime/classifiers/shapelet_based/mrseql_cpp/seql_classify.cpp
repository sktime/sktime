/*
 * seql_classify.cpp
 *
 *  Created on: 19 Jul 2016
 *      Author: thachln
 */




/*
 * seql_classify.h
 *
 *  Created on: 7 Jul 2016
 *      Author: thachln
 */



#include <limits>
#include <vector>
#include <string>
#include <map>
#include "mmap.h"
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>
//#include "common.h"
#include "common_string_symbol.h"
#include "darts.h"
#include "sys/time.h"
#include "seql_classify.h"

static inline char *read_ptr (char **ptr, size_t size)
{
	char *r = *ptr;
	*ptr += size;
	return r;
}

template <class T> static inline void read_static (char **ptr, T& value)
{
	char *r = read_ptr (ptr, sizeof (T));
	memcpy (&value, r, sizeof (T));
}

template <typename T1, typename T2>
struct pair_2nd_cmp_alt: public std::binary_function<bool, T1, T2> {
	bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
	{
		return x1.second > x2.second;
	}
};





void SEQLClassifier::project (std::string prefix,
		unsigned int pos,
		size_t trie_pos,
		size_t str_pos,
		bool token_type)
{
	if (pos == doc.size() - 1) return;

	// Check traversal with both the next actual unigram in the doc and the wildcard *.
	string next_unigrams[2];
	next_unigrams[0] = doc[pos + 1].key();
	next_unigrams[1] = "*";

	for (int i = 0; i < 2; ++i) {

		string next_unigram = next_unigrams[i];
		std::string item;
		if (!token_type) { //word-level token
			item = prefix + " " + next_unigram;
		} else { // char-level token
			item = prefix + next_unigram;
		}
		//cout << "\nitem: " << item.c_str();
		size_t new_trie_pos = trie_pos;
		size_t new_str_pos  = str_pos;
		int id = da.traverse (item.c_str(), new_trie_pos, new_str_pos);
		//cout <<"\nid: " << id;

		//if (id == -2) return;
		if (id == -2) {
			if (i == 0) continue;
			else return;
		}
		if (id >= 0) {
			if (userule) {
				//cout << "\nnew rule: " << item;
				rules.insert (std::make_pair (item, alpha[id]));
				rules_and_ids.insert (std::make_pair (item, id));
			}
			result.push_back (id);
		}
		project (item, pos + 1, new_trie_pos, new_str_pos, token_type);
	}
}





double SEQLClassifier::getBias() {
	return bias;
}

int SEQLClassifier::getOOVDocs() {
	return oov_docs;
}

void SEQLClassifier::setRule(bool t)
{
	userule = t;
}

bool SEQLClassifier::open (const char *file, double threshold)
{
	if (! mmap.open (file)) return false;

	char *ptr = mmap.begin ();
	unsigned int size = 0;
	read_static<unsigned int>(&ptr, size);
	da.set_array (ptr);
	ptr += size;
	read_static<double>(&ptr, bias);    // this bias from the model file is not used for classif; it is automatically obtained by summing
	// up the features of the model and it is used for info only
	bias = -threshold;  //set bias to minus user-provided-thereshold

	alpha = (double *)ptr;

	return true;
}

// Compute the area under the ROC curve.
double SEQLClassifier::calcROC( std::vector< std::pair<double, int> >& forROC )
{
	//std::sort( forROC.begin(), forROC.end() );
	double area = 0;
	double x=0, xbreak=0;
	double y=0, ybreak=0;
	double prevscore = - numeric_limits<double>::infinity();
	for( vector< pair<double, int> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
	{
		double score = ritr->first;
		int label = ritr->second;
		//cout << "\nscore: " << score << " label: " << label;
		if( score != prevscore ) {
			//cout << "\nx: " << x << " xbreak: " << xbreak << " y: " << y << " ybreak: " << ybreak;
			area += (x-xbreak)*(y+ybreak)/2.0;
			//cout << "\narea: " << area;
			xbreak = x;
			ybreak = y;
			prevscore = score;
		}
		if( label > 0)  y ++;
		else     x ++;
	}
	area += (x-xbreak)*(y+ybreak)/2.0; //the last bin
	if( 0==y || x==0 )   area = 0.0;   // degenerate case
	else        area = 100.0 * area /( x*y );
	//cout << "\narea: " << area;
	return area;
}

// Compute the area under the ROC50 curve.
// Fixes the number of negatives to 50.
// Stop computing curve after seeing 50 negatives.
double SEQLClassifier::calcROC50( std::vector< std::pair<double, int> >& forROC )
{
	//std::sort( forROC.begin(), forROC.end() );
	double area50 = 0;
	double x=0, xbreak=0;
	double y=0, ybreak=0;
	double prevscore = - numeric_limits<double>::infinity();
	for( vector< pair<double, int> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
	{
		double score = ritr->first;
		int label = ritr->second;

		if( score != prevscore && x < 50) {
			area50 += (x-xbreak)*(y+ybreak)/2.0;
			xbreak = x;
			ybreak = y;
			prevscore = score;
		}
		if( label > 0)  y ++;
		else if (x < 50) x ++;
	}
	area50 += (x-xbreak)*(y+ybreak)/2.0; //the last bin
	if( 0==y || x==0 )   area50 = 0.0;   // degenerate case
	else        area50 = 100.0 * area50 /( 50*y );
	return area50;
}

double SEQLClassifier::classify (const char *line, bool token_type)
{
	result.clear ();
	doc.clear ();
	rules.clear ();
	double r = bias;

	// Prepare instance as a vector of string_symbol
	str2node (line, doc, token_type);

	for (unsigned int i = 0; i < doc.size(); ++i) {
		std::string item = doc[i].key();
		int id;
		da.exactMatchSearch (item.c_str(), id);
		//int id = da.exactMatchSearch (doc[i].key().c_str());
		if (id == -2) continue;
		if (id >= 0) {
			if (userule) {
				rules.insert (std::make_pair (doc[i].key(), alpha[id]));
				rules_and_ids.insert (std::make_pair (doc[i].key(), id));
			}
			result.push_back (id);
		}
		project (doc[i].key(), i, 0, 0, token_type);
	}

	std::sort (result.begin(), result.end());

	// Binary frequencies, erase the duplicate feature ids, features count only once.
	result.erase (std::unique (result.begin(), result.end()), result.end());

	if (result.size() == 0) {
		if (userule)
			cout << "\n Test doc out of vocabulary\n";
		oov_docs++;
	}
	for (unsigned int i = 0; i < result.size(); ++i) r += alpha[result[i]];

	return r;
}

double SEQLClassifier::classifyBOSS (const char *line, bool token_type, int sub_len)
{
	result.clear ();
	//doc.clear ();
	rules.clear ();
	double r = bias;

	// Prepare instance as a vector of string_symbol
	// @THACH: split line into subsequences


	int subi = 0;
	char sline[sub_len + 1];
	while(subi < strlen(line)){
		doc.clear ();
		memcpy(sline, &line[subi], sub_len);
		sline[sub_len] = '\0';
		// cout << sline << endl;
		str2node (sline, doc, token_type);

		for (unsigned int i = 0; i < doc.size(); ++i) {
			std::string item = doc[i].key();
			int id;
			da.exactMatchSearch (item.c_str(), id);
			//int id = da.exactMatchSearch (doc[i].key().c_str());
			if (id == -2) continue;
			if (id >= 0) {
				if (userule) {
					rules.insert (std::make_pair (doc[i].key(), alpha[id]));
					rules_and_ids.insert (std::make_pair (doc[i].key(), id));
				}
				result.push_back (id);
			}
			project (doc[i].key(), i, 0, 0, token_type);
		}

		subi += sub_len + 1;
	}


	std::sort (result.begin(), result.end());

	// Binary frequencies, erase the duplicate feature ids, features count only once.
	result.erase (std::unique (result.begin(), result.end()), result.end());

	if (result.size() == 0) {
		if (userule)
			cout << "\n Test doc out of vocabulary\n";
		oov_docs++;
	}
	for (unsigned int i = 0; i < result.size(); ++i) {

		r += alpha[result[i]];
	}

	return r;
}

std::ostream &SEQLClassifier::printRules (std::ostream &os)
{
	std::vector <std::pair <std::string, double> > tmp;

	for (std::map <std::string, double>::iterator it = rules.begin();
			it != rules.end(); ++it)
		tmp.push_back (std::make_pair (it->first,  it->second));

	std::sort (tmp.begin(), tmp.end(), pair_2nd_cmp_alt<std::string, double>());
	os << "\nrule: " << bias << " __DFAULT__" << std::endl;

	//    for (std::vector <std::pair <std::string, double> >::iterator it = tmp.begin();
	//   it != tmp.end(); ++it)
	for (std::map <std::string, double>::iterator it = rules.begin();
			it != rules.end(); ++it)
		//os << "rule: " << rules_and_ids[it->first] << " " << it->second << " " << it->first << std::endl;
		os << "rule: " << it->first << " " << it->second << std::endl;

	return os;
}

std::ostream &SEQLClassifier::printIds (std::ostream &os) {
	for (std::map <std::string, int>::iterator it = rules_and_ids.begin(); it != rules_and_ids.end(); ++it)
		os << (it->second + 1) << ":1.0 ";
	os << "\n";

	return os;
}






