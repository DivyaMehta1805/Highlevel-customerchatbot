from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrape_it import GeneralSpider
import asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class URLInput(BaseModel):
    url: HttpUrl
class CrawlerService:
    def __init__(self):
        self.process = CrawlerProcess(get_project_settings())

    async def run_crawler(self, start_url):
        try:
            def crawl_wrapper():
                self.process.crawl(GeneralSpider, start_url=start_url)
                self.process.start()

            await asyncio.get_event_loop().run_in_executor(None, crawl_wrapper)
            return True
        except Exception as e:
            print(f"Crawling error: {str(e)}")
            return False

crawler_service = CrawlerService()

@app.post("/crawl")
async def crawl(url_input: URLInput):
    try:
        success = await crawler_service.run_crawler(str(url_input.url))
        if success:
            return {"message": "Crawl completed successfully", "status": "success"}
        else:
            return {"message": "Crawl failed", "status": "failure"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
