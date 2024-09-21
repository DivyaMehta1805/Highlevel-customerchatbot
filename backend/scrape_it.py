import scrapy
from scrapy.http import Request
from scrapy.crawler import CrawlerProcess
from itemadapter import ItemAdapter
from collections import deque
from scrapy.exceptions import CloseSpider
import json
class GeneralSpider(scrapy.Spider):
    name = 'general'
    
    def __init__(self, start_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.crawled_urls = set()  # To store unique URLs
        self.queue = deque([start_url])  # Queue to manage URLs
        self.max_urls = 100  # Limit to 100 URLs
        self.crawled_data = []  # To store crawled data

    def parse(self, response):
        if len(self.crawled_urls) >= self.max_urls:
            raise CloseSpider(reason="Reached URL limit of 100")

        # Extract all links from the page and add them to the queue
        page_links = response.css('a::attr(href)').getall()

        for link in page_links:
            # Normalize the link and check if it's already crawled
            absolute_url = response.urljoin(link)
            if absolute_url not in self.crawled_urls and len(self.crawled_urls) < self.max_urls:
                self.queue.append(absolute_url)

        # Mark the current page as crawled
        self.crawled_urls.add(response.url)

        # Store the page content or data
        data = {
            'url': response.url,
            'content': response.text,  # Storing the page content
        }
        self.crawled_data.append(data)

        # Save data to JSON file after every 10 pages
        if len(self.crawled_data) % 10 == 0:
            self.save_data()

        # Yield the data for Scrapy's processing
        yield data

        # Crawl next URLs from the queue
        if self.queue:
            next_url = self.queue.popleft()
            yield Request(next_url, callback=self.parse)

    def save_data(self):
        with open('crawled_data.json', 'w') as f:
            json.dump(self.crawled_data, f, indent=4)
        self.logger.info(f"Saved {len(self.crawled_data)} pages to crawled_data.json")

    def closed(self, reason):
        self.save_data()  # Save any remaining data when spider closes