from icrawler.builtin import BingImageCrawler

categories = {
    "cats": "domestic cat",
    "dogs": "domestic dog",
    "birds": "colorful bird",
    "cars": "modern sedan car",
    "flowers": "rose flower"
}

for folder, keyword in categories.items():
    print(f"\nDownloading {folder} images...")

    crawler = BingImageCrawler(storage={"root_dir": f"data/dataset/{folder}"})

    crawler.crawl(
        keyword=keyword,
        max_num=200
    )

print("\nDataset downloaded successfully!")