require('twitter')
require 'csv'    

client = Twitter::REST::Client.new do |config|
  config.consumer_key    = ""
  config.consumer_secret = ""
end

def collect_with_max_id(collection=[], max_id=nil, &block)
  response = yield(max_id)
  collection += response
  response.empty? ? collection.flatten : collect_with_max_id(collection, response.last.id - 1, &block)
end

def client.get_all_tweets(user)
  collect_with_max_id do |max_id|
    options = {count: 200, include_rts: false}
    options[:max_id] = max_id unless max_id.nil?
	begin
		user_timeline(user, options)
	rescue Twitter::Error::NotFound, Twitter::Error::Unauthorized => e
	rescue Twitter::Error::TooManyRequests => error
		# NOTE: Your process could go to sleep for up to 15 minutes but if you
		# retry any sooner, it will almost certainly fail with the same exception.
		sleep error.rate_limit.reset_in + 1
		retry
	end
  end
end

first = true
Dir["./classification_processed/bots/*"].each do |file| 
puts file
	csv_text = File.read(file)
	csv = CSV.parse(csv_text, :headers => true)

	CSV.open("tweets-bots.csv", "ab") do |csv_out|
		if first
			csv_out << ["user","text"]
			first = false
		end
		csv.each do |row|
			puts row[0]
			client.get_all_tweets(row[0]).each { |x| 
			if x.lang=="en" 
				csv_out << [row[0],x.text] 
			end
			}
		end
	end
end
