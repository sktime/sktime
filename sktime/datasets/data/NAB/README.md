NAB Data Corpus
---

Data are ordered, timestamped, single-valued metrics. All data files contain anomalies, unless otherwise noted.


### Real data
- realAWSCloudwatch/

	AWS server metrics as collected by the AmazonCloudwatch service. Example metrics include CPU Utilization, Network Bytes In, and Disk Read Bytes.

- realAdExchange/

	Online advertisement clicking rates, where the metrics are cost-per-click (CPC) and cost per thousand impressions (CPM). One of the files is normal, without anomalies.

- realKnownCause/

	This is data for which we know the anomaly causes; no hand labeling.

	- ambient_temperature_system_failure.csv: The ambient temperature in an office
	setting.
	- cpu_utilization_asg_misconfiguration.csv: From Amazon Web Services (AWS)
	monitoring CPU usage – i.e. average CPU usage across a given cluster. When
	usage is high, AWS spins up a new machine, and uses fewer machines when usage
	is low.
	- ec2_request_latency_system_failure.csv: CPU usage data from a server in
	Amazon's East Coast datacenter. The dataset ends with complete system failure
	resulting from a documented failure of AWS API servers. There's an interesting
	story behind this data in the [Numenta
	blog](http://numenta.com/blog/anomaly-of-the-week.html).
	- machine_temperature_system_failure.csv: Temperature sensor data of an
	internal component of a large, industrial mahcine. The first anomaly is a
	planned shutdown of the machine. The second anomaly is difficult to detect and
	directly led to the third anomaly, a catastrophic failure of the machine.
	- nyc_taxi.csv: Number of NYC taxi passengers, where the five anomalies occur
	during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow
	storm. The raw data is from the [NYC Taxi and Limousine Commission](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).
	The data file included here consists of aggregating the total number of
	taxi passengers into 30 minute buckets.
	- rogue_agent_key_hold.csv: Timing the key holds for several users of a
	computer, where the anomalies represent a change in the user.
	- rogue_agent_key_updown.csv: Timing the key strokes for several users of a
	computer, where the anomalies represent a change in the user.

- realTraffic/

	Real time traffic data from the Twin Cities Metro area in Minnesota, collected
	by the
	[Minnesota Department of Transportation](http://www.dot.state.mn.us/tmc/trafficinfo/developers.html).
	Included metrics include occupancy, speed, and travel time from specific
	sensors.

- realTweets/

	A collection of Twitter mentions of large publicly-traded companies
	such as Google and IBM. The metric value represents the number of mentions
	for a given ticker symbol every 5 minutes.


### Artificial data

- artificialNoAnomaly/

	Artificially-generated data without any anomalies.

- artificialWithAnomaly/

	Artificially-generated data with varying types of anomalies.
