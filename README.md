## Twitter Lexica

This script predicts the age and gender using their recent tweets. The prediction model used is from the [lexica](https://github.com/wwbp/lexica) project developed by the World Well-Being Project.

## Install

Clone this repo and install the required python module. The script was developed using python 3.7.

```
git clone https://www.github.com/zhuchcn/twitter-lexica.git
pip install requirements.txt
```

## Twitter API

A valid Twitter API key is required in order to retrieve user's tweets from twitter. The API keys must be put into the `.env` file. Replace the XXX with your own twitter api keys.

```
CONSUMER_KEY="XXX"
CONSUMER_SECRETE="XXX"
ACCESS_KEY="XXX"
ACCESS_SECRETE="XXX"
```

## Usage

To predict a single user:

```
python predict_twitter.py -s @realDonaldTrump
```

To predict a list of user, use a txt file with each user name in a row.
```
python predict_twitter.py -i users.txt -o predict.txt
```

To save all tweets used, use a directory
```
python predict_twitter.py -i users.txt -o predict.txt -d user_twitters
```

To save the output log
```
python predict_twitter.py -i test.txt -o predict.txt -d user_twitters > log.txt
```

To get help
```
python predict_twitter.py --help
```