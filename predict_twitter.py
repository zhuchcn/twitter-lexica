import tweepy
import re
import os
from pandas import read_csv
import nltk
import string
import re
import argparse
import csv
from dotenv import load_dotenv


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secrete = os.environ.get("CONSUMER_SECRETE")
access_key = os.environ.get("ACCESS_KEY")
access_secrete = os.environ.get("ACCESS_SECRETE")

auth = tweepy.OAuthHandler(consumer_key, consumer_secrete)
auth.set_access_token(access_key, access_secrete)
api = tweepy.API(auth)

class Lexica():
    def __init__(self):
        age = read_csv("lexica/emnlp14age.csv")
        self.age = {row["term"]: row["weight"] for index, row in age.iterrows()}
        gender = read_csv("lexica/emnlp14gender.csv")
        self.gender = {row["term"]: row["weight"] for index, row in gender.iterrows()}

lexica = Lexica()

class TwitterLexica():
    def __init__(self, screen_name, max_tweets=200, count=200):
        self.lexica = lexica
        self.user_name = screen_name
        self.tweets = self.get_tweets(screen_name, max_tweets, count)
        
    def __repr__(self):
        return f"<TwitterLexica: {self.user_name}>"

    def get_tweets(self, screen_name, max_tweets, count=200):
        max_request = 10
        alltweets = []

        new_tweets = self.fetch_tweets(screen_name, count)
        if len(new_tweets) == 0:
            print(f"user {screen_name} has 0 tweets")
            return alltweets
        
        oldest = new_tweets[-1][0] - 1
        new_tweets = [tweet for tweet in new_tweets \
                    if not tweet[2].startswith('RT @')]
        alltweets.extend(new_tweets)

        i = 0
        while len(new_tweets) > 0:
            new_tweets = self.fetch_tweets(screen_name, count, max_id=oldest)    
            if len(new_tweets) == 0:
                print(f"user {screen_name} has 0 tweets")
                return alltweets

            oldest = new_tweets[-1][0] - 1
            new_tweets = [tweet for tweet in new_tweets \
                        if not tweet[2].startswith('RT @')]
            alltweets.extend(new_tweets)
            
            if(len(alltweets) > max_tweets):
                alltweets = alltweets[:max_tweets]
                return alltweets
            
            i += 1
            if i >= 10:
                return alltweets

    @staticmethod
    def fetch_tweets(screen_name, count, **kwargs):
        new_tweets = api.user_timeline(
            screen_name=screen_name,
            count=count,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True,
            **kwargs
        )
        new_tweets = [
            (tweet.id, tweet.created_at, tweet.text) \
                for tweet in new_tweets
        ]
        return new_tweets
    
    def get_freq(self):
        strings = [tweet[2] for tweet in self.tweets]
        strings = [re.sub("http?:\/\/.*[\r\n]*", "", s) for s in strings]
        strings = [re.sub("\/\/.*[\r\n]*", "", s) for s in strings]
        strings = [re.sub("``", "", s) for s in strings]
        words = nltk.word_tokenize(''.join(strings))
        punctuations = list(string.punctuation)
        punctuations.append("''")
        words = [word for word in words if not word in punctuations]
        words = [word.lower() for word in words]
        freqs = nltk.FreqDist(words)
        return freqs
        
    
    def predict(self):
        freqs = self.get_freq()
        total_freq = sum([freqs[key] for key in freqs])

        age = self.lexica.age["_intercept"]
        for key, val in freqs.items():
            weight = self.lexica.age.get(key)
            if not weight:
                continue
            age += weight * val / total_freq

        gender = self.lexica.gender["_intercept"]
        for key, val in freqs.items():
            weight = self.lexica.gender.get(key)
            if not weight:
                continue
            gender += weight * val / total_freq

        return(age, gender)
    
    def save_twitters(self, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        path = os.path.join(output_dir, f"{self.user_name}.csv")
        with open(path,"w") as f: 
            writer = csv.writer(f)
            writer.writerow(["id","created_at","text"])
            writer.writerows(self.tweets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--screen-name", type=str, default=None,
        help="Twitter user name"
    )
    parser.add_argument(
        "-i", "--input-file", type=str, default=None,
        help="TXT file with each twitter user name on each line"
    )
    parser.add_argument(
        '-o', '--output-file', type=str, default=None,
        help="The output file for predicted Age and Gender"
    )
    parser.add_argument(
        '-d', '--output-dir', default=None,
        help="""
        The diractory to save user twitters. Twitters won't save if this is not 
        given
        """
    )
    parser.add_argument(
        '-m', '--max-tweets', default=200,
        help="Max number of must recent twitters to use."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.screen_name:
        if args.input_file:
            print("The input file is ignored")
        
        try:
            tl = TwitterLexica(args.screen_name, args.max_tweets)
            if args.output_dir:
                tl.save_twitters(args.output_dir)
            age, gender = tl.predict() if tl.tweets else (None, None)
        except tweepy.TweepError as e:
            if e.api_code == 34:
                print(f"user {args['screen_name']} was not found")
            else:
                print(e)
            age, gender = None, None

        if args.output_file:
            num_tweets = len(tl.tweets) if tl.tweets else 0
            with open(args.output_file, "w") as fh:
                fh.write("user name\tnum tweets\tage\tgender\n")
                fh.write(f"{args.screen_name}\t{num_tweets}\t{age}\t{gender}\n")
        else:
            print(f"""Username: {tl.user_name}
Lexica prediction base on {len(tl.tweets) if tl.tweets else 0} tweets:
    Age: {age},
    Gender: {gender}""")

    elif args.input_file:
        if not args.output_file:
            raise ValueError("--output-file must not be None")
        
        with open(args.output_file, "w") as fh:
            fh.write("user name\tnum tweets\tage\tgender\n")
        
        with open(args.input_file,"r") as fh:
            for line in fh:
                screen_name = line.rstrip()
                screen_name = re.sub('^"|"$', "", screen_name)
                screen_name = re.sub("^'|'$", "", screen_name)
                
                if not screen_name.startswith("@"):
                    continue
                try:
                    tl = TwitterLexica(screen_name, args.max_tweets)
                    if args.output_dir:
                        tl.save_twitters(args.output_dir)

                    age, gender = tl.predict() if tl.tweets else (None, None)

                except tweepy.TweepError as e:
                    if e.api_code == 34:
                        print(f"user {screen_name} was not found")
                    else:
                        print(e)
                    age, gender = None, None

                with open(args.output_file, "a") as fh:
                    num_tweets = len(tl.tweets) if tl.tweets else 0
                    fh.write(f"{screen_name}\t{num_tweets}\t{age}\t{gender}\n")
                print(f"user {screen_name} was saved")
                if args.output_dir:
                    tl.save_twitters(args.output_dir)

if __name__ == "__main__":          
    main()            
