/*
 * seql_learn.h
 *
 *  Created on: 6 Jul 2016
 *      Author: thachln
 */

#ifndef SEQL_LEARN_H_
#define SEQL_LEARN_H_


//#include "sax_converter.h"
#include <vector>
#include <string>
#include <map>
#include <set>
// #include "sys/time.h" # not working with VSC++
#include <list>
#include "SNode.h"
// #include <unistd.h>
//using namespace std;


class SeqLearner {

private:



    // Best ngram rule.
    struct rule_t {
        // Gradient value for this ngram.
        double gradient;
        // Length of ngram.
        unsigned int size;
        // Ngram label.
        std::string  ngram;
        // Ngram support, e.g. docids where it occurs in the collection.
        std::vector <unsigned int> loc;
        friend bool operator < (const rule_t &r1, const rule_t &r2)
        {
            return r1.ngram < r2.ngram;
        }
    };

    // Flag to control whether data has been read to memory. Sometimes client code may want to use external read function
    bool data_read;
    // Entire collection of documents, each doc represented as a string.
    // The collection is a vector of strings.
    std::vector < std::string > transaction;
    // True classes.
    std::vector < double > y;
    // The fraction: 1 / 1 + exp^(yi*beta^t*xi) in the gradient computation.
    std::vector < long double > exp_fraction;
    // Per document, sum of best beta weights, beta^t * xi = sum_{j best beta coord} gradient_j
    std::vector < double > sum_best_beta;
    // The scalar product obtained with the optimal beta according to the line search for best step size.
    std::vector < double > sum_best_beta_opt;
    // Regularized loss function: loss + C * elasticnet_reg
    // SLR loss function: log(1+exp(-yi*beta^t*xi))
    // Squared Hinge SVM loss function: sum_{i|1-yi*beta^t*xi > 0} (1 - yi*beta^t*xi)^2
    // Squared error: (yi-beta^t*xi)^2
    long double loss;
    long double old_loss; //keep loss in previous iteration for checking convergence

    std::map <std::string, double> features_cache;
    std::map<std::string, double>::iterator features_it;

    // PARAMETERS
    // Objective function. For now choice between logistic regression, l2 (Squared Hinge Loss) and squared error loss.
    unsigned int objective;
    // Regularizer value.
    double C;
    // Weight on L1 vs L2 regularizer.
    double alpha;
    // Max length for an ngram.
    unsigned int maxpat;
    // Min length for an ngram.
    unsigned int minpat;
    // Min supoort for an ngram.
    unsigned int minsup;

    // The sum of squared values of all non-zero beta_j.
    double sum_squared_betas;

    // The sum of abs values of all non-zero beta_j.
    double sum_abs_betas;

    std::set <std::string> single_node_minsup_cache;

    // Current suboptimal gradient.
    double       tau;

    // Total number of times the pruning condition is checked
    unsigned int total;
    // Total number of times the pruning condition is satisfied.
    unsigned int pruned;
    // Total number of times the best rule is updated.
    unsigned int rewritten;

    // Convergence threshold on aggregated change in score predictions.
    // Used to automatically set the number of optimisation iterations.
    double convergence_threshold;

    // Verbosity level: 0 - print no information,
    //                  1 - print profiling information,
    //                  2 - print statistics on model and obj-fct-value per iteration
    //                  > 2 - print details about search for best n-gram and pruning process
    int verbosity;

    // Traversal strategy: BFS or DFS.
    bool traversal_strategy;

    // Profiling variables.
    // struct timeval t;
    // struct timeval t_origin;
    // struct timeval t_start_iter;
    //long double LDBL_MAX = numeric_limits<long double>::max();

    // lines ignored by SEQL learner
    std::set<int> skip_items;


    // Read the input training documents, "true_class document" per line.
    // A line in the training file can be: "+1 a b c"
    bool read (const char *filename);
    // For current ngram, compute the gradient value and check prunning conditions.
    // Update the current optimal ngram.
    bool can_prune_and_update_rule (rule_t& rule, SNode *space, unsigned int size);
    // Try to grow the ngram to next level, and prune the appropriate extensions.
    // The growth is done breadth-first, e.g. grow all unigrams to bi-grams, than all bi-grams to tri-grams, etc.
    void span_bfs (rule_t& rule, SNode *space, std::vector<SNode *>& new_space, unsigned int size);

    void createCandidatesExpansions(SNode* space, std::map<std::string, SNode>& candidates);

    // Try to grow the ngram to next level, and prune the appropriate extensions.
    // The growth is done deapth-first rather than breadth-first, e.g. grow each candidate to its longest unpruned sequence
    void span_dfs (rule_t& rule, SNode *space, unsigned int size);
    // Line search method. Search for step size that minimizes loss.
    // Compute loss in middle point of range, beta_n1, and
    // for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
    // Compare the loss for the 3 points, and choose range of 3 points
    // which contains the minimum. Repeat until the range spanned by the 3 points is small enough,
    // e.g. the range approximates well the vector where the loss function is minimized.
    // Return the middle point of the best range.
    void find_best_range(std::vector<double>& sum_best_beta_n0, std::vector<double>& sum_best_beta_n1, std::vector<double>& sum_best_beta_n2,
    		std::vector<double>& sum_best_beta_mid_n0_n1, std::vector<double>& sum_best_beta_mid_n1_n2,
                         rule_t& rule, std::vector<double>* sum_best_beta_opt);

    // Line search method. Binary search for optimal step size. Calls find_best_range(...).
    // sum_best_beta keeps track of the scalar product beta_best^t*xi for each doc xi.
    // Instead of working with the new weight vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n)
    // we work directly with the scalar product.
    // We output the sum_best_beta_opt which contains the scalar poduct of the optimal beta found, by searching for the optimal
    // epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
    // epsilon is the starting value
    // rule contains info about the gradient at the current iteration
    void binary_line_search(rule_t& rule, std::vector<double>* sum_best_beta_opt);
    // Searches the space of all subsequences for the ngram with the ngram with the maximal abolute gradient and saves it in rule
    rule_t findBestNgram(rule_t& rule ,std::vector <SNode*>& old_space, std::vector<SNode*>& new_space, std::map<std::string, SNode>& seed);

public:

    SeqLearner();

    void add_skips_items(int item);

    //void clear_skip_list();

    // public function for read in case client code want to read data from memory
    bool external_read (std::vector<std::string>& data);
    bool external_read (std::vector < std::string >& _transaction, std::vector < double >& _y);


    int run (const char *in,
              const char *out,
			  //unsigned int _sax_window_size,
			  //unsigned int _sax_word_length,
			  //unsigned int _sax_alphabet_size,
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
              int _verbosity);

    int run_internal (
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
                  int _verbosity);

    void prepareInvertedIndex (std::map<std::string, SNode>& seed);

    void deleteTree(std::map<std::string, SNode>& seed);

    void deleteUndersupportedUnigrams(std::map<std::string, SNode>& seed);

    long double computeLossTerm(const double& beta, const double &y);

    long double computeLossTerm(const double& beta, const double &y, long double &exp_fraction);

    void computeLoss(long double &loss, const std::vector<double>& beta);

    void computeLoss(long double &loss, const std::vector<double>& beta,
                     double &sum_abs_scalar_prod_diff, double &sum_abs_scalar_prod , std::vector<double long>& exp_fraction);

    void computeLoss(long double &loss, const std::vector<double>& beta,
                     double &sum_abs_scalar_prod_diff, double &sum_abs_scalar_prod );

    bool setup(const char *in, const char *out, std::ofstream& os);
    bool setup_internal();



};





#endif /* SEQL_LEARN_H_ */
