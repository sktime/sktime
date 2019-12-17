/*
 * sfa_seql.cpp
 *
 *  Created on: 5 Dec 2017
 *      Author: thachln
 */

#include "mr_seql_classifier.cpp"

int main(int argc, char **argv){



	string train_data;
	string test_data;
	string output;
	bool multiclass = false;




	int opt;
	while ((opt = getopt(argc, argv, "t:T:o:m:")) != -1) {
		switch(opt) {
		case 't':
			train_data = string (optarg);
			break;
		case 'T':
			test_data = string (optarg);
			break;
		case 'o':
			output = string (optarg);
			break;
		case 'm':
			multiclass = atoi(optarg);
			break;
		default:
			std::cout << "Usage: " << argv[0] << std::endl;
			return -1;
		}
	}

	ensemble_seql(train_data,test_data,output,true);

}
