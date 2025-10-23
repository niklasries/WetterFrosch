# kachelmann-scraper
# python your_script_name.py [START_TIME] [END_TIME] [OPTIONS]
# python wetter_scraper.py 20250927-2200 20230101-0000 --output-dir "/wetter/input/WetterDaten/"

import re
import os
from datetime import datetime, timedelta
import time
import argparse
import random
import base64 

# --- Selenium Imports ---
from selenium import webdriver # type: ignore
from selenium.webdriver.chrome.service import Service as ChromeService # type: ignore 
from webdriver_manager.chrome import ChromeDriverManager # type: ignore
from selenium.webdriver.chrome.options import Options  # type: ignore

# --- Configuration ---
SAT_PAGE_URL_TEMPLATE = "https://kachelmannwetter.com/de/sat/deutschland/satellit-nature-15min/{timestamp}z.html"
RADAR_PAGE_URL_TEMPLATE = "https://kachelmannwetter.com/de/radar-standard/deutschland/{timestamp}z.html"

SAT_IMAGE_URL_PATTERN = r'(https://img\d+\.kachelmannwetter\.com/images/data/cache/sat/sat_[\w\d/_\-\.]+\.jpg)'
RADAR_IMAGE_URL_PATTERN = r'(https://img\d+\.kachelmannwetter\.com/images/data/cache/radar/radar_[\w\d/_\-\.]+\.png)'


def setup_browser():
    """Sets up and returns a headless Selenium Chrome browser instance."""
    print("Setting up headless browser...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_script_timeout(45)
    print("Browser setup complete.")
    return driver

# handle retries with exponential backoff
def safe_get(driver, url, max_retries=5, initial_delay=5):
    """
    Attempts to navigate to a URL, handling 429 errors with exponential backoff.
    Returns True on success, False on failure after all retries.
    """
    retries = 0
    while retries < max_retries:
        try:
            driver.get(url)
            # A simple way to check for a 429 error page
            if "429" in driver.title or "Too Many Requests" in driver.page_source:
                raise ValueError("429 Too Many Requests detected")
            return True # Success
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"  [FATAL] Failed to load {url} after {max_retries} retries. Error: {e}")
                return False
            
            # Exponential backoff calculation with jitter
            backoff_time = initial_delay * (2 ** (retries - 1)) + random.uniform(0, 1)
            print(f"  [Warning] Request failed (Attempt {retries}/{max_retries}). Retrying in {backoff_time:.2f} seconds...")
            time.sleep(backoff_time)
    return False

def get_html(driver, url):
    """Navigates to a URL using safe_get and returns the page source."""
    if safe_get(driver, url):
        time.sleep(2) # Still give the page a moment to render JS if needed
        return driver.page_source
    else:
        print(f"  [Error] Browser could not load page {url} after multiple retries.")
        return None

def extract_image_url(html_content, pattern):
    """Extracts an image URL from HTML content using a regex pattern."""
    if not html_content: return None
    match = re.search(pattern, html_content)
    return match.group(1) if match else None

def dl_img_direct(driver, image_url, save_path):
    """Downloads the raw image file using a JavaScript fetch and Base64 encoding."""
    if not image_url: return False
    print(f"  Downloading (direct): {os.path.basename(save_path)}")
    js_script = """
        var url = arguments[0], callback = arguments[1];
        fetch(url).then(r => r.blob()).then(b => {
            var reader = new FileReader();
            reader.onload = () => callback(reader.result);
            reader.readAsDataURL(b);
        }).catch(e => callback('Error: ' + e));
    """
    try:
        base64_data = driver.execute_async_script(js_script, image_url)
        if base64_data.startswith('Error:'):
            print(f"  [Error] JavaScript fetch failed: {base64_data}")
            return False
        header, encoded = base64_data.split(',', 1)
        image_data = base64.b64decode(encoded)
        with open(save_path, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"  [Error] Failed to download or save {image_url}: {e}")
        return False

def dl_img(driver, image_url, save_path):
    """Downloads an image using safe_get and screenshotting the element."""
    if not image_url:
        print("  [Error] No image URL provided to download.")
        return False

    print(f"  Downloading: {os.path.basename(save_path)}")
    if safe_get(driver, image_url):
        try:
            image_element = driver.find_element("tag name", "img")
            image_data = image_element.screenshot_as_png
            with open(save_path, 'wb') as f:
                f.write(image_data)
            return True
        except Exception as e:
            print(f"  [Error] Page loaded, but failed to screenshot/save {image_url}: {e}")
            return False
    else:
        print(f"  [Error] Failed to download image from {image_url} after multiple retries.")
        return False


def main():
    parser = argparse.ArgumentParser(description="A resilient, Selenium-based web crawler for Kachelmannwetter images.")
    parser.add_argument("start_time", help="The starting timestamp (most recent date) in YYYYMMDD-HHmm format.")
    parser.add_argument("end_time", help="The ending timestamp (oldest date, inclusive) in YYYYMMDD-HHmm format.")
    parser.add_argument("--output-dir", default="weather_images", help="Directory to save images.")
    parser.add_argument("--method", default="direct", choices=['direct', 'screenshot'], 
                        help="Download method: 'direct' for clean files (default), 'screenshot' for rendered images.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Images will be saved in '{args.output_dir}/'")
    print(f"Using download method: {args.method}")

    try:
        start_dt = datetime.strptime(args.start_time, "%Y%m%d-%H%M")
        end_dt = datetime.strptime(args.end_time, "%Y%m%d-%H%M")
    except ValueError:
        print("Error: Invalid date format. Please use YYYYMMDD-HHmm for both start and end times.")
        return

    if start_dt < end_dt:
        print(f"Error: The start_time ({args.start_time}) must be later than or equal to the end_time ({args.end_time}).")
        return

    driver = setup_browser()
    
    current_dt = start_dt
    
    try:
        while current_dt >= end_dt:
            timestamp_str = current_dt.strftime("%Y%m%d-%H%M")

            #skip if not a 15 min
            if current_dt.minute % 15 != 0:
                current_dt -= timedelta(minutes=5)
                continue

            if args.method == 'direct':
                sat_filename = f"sat_{timestamp_str}.png"
                radar_filename = f"radar_{timestamp_str}.png"
            else: # screenshot method always produces PNGs
                sat_filename = f"sat_{timestamp_str}.png"
                radar_filename = f"radar_{timestamp_str}.png"
            sat_save_path = os.path.join(args.output_dir, sat_filename)
            radar_save_path = os.path.join(args.output_dir, radar_filename)

            if os.path.exists(sat_save_path) and os.path.exists(radar_save_path):
                #print(f"\nSkipping timestamp: {timestamp_str} (files already exist)")
                current_dt -= timedelta(minutes=5)
                continue

            print(f"\nProcessing timestamp: {timestamp_str}")

            download_function = dl_img_direct if args.method == 'direct' else dl_img

            sat_page_url = SAT_PAGE_URL_TEMPLATE.format(timestamp=timestamp_str)
            sat_html = get_html(driver, sat_page_url)
            if sat_html:
                sat_image_url = extract_image_url(sat_html, SAT_IMAGE_URL_PATTERN)
                if sat_image_url:
                    download_function(driver, sat_image_url, sat_save_path)
                else:
                    print("  Could not process satellite image for this timestamp.")

            radar_page_url = RADAR_PAGE_URL_TEMPLATE.format(timestamp=timestamp_str)
            radar_html = get_html(driver, radar_page_url)
            if radar_html:
                radar_image_url = extract_image_url(radar_html, RADAR_IMAGE_URL_PATTERN)
                if radar_image_url:
                    download_function(driver, radar_image_url, radar_save_path)
                else:
                     print("  Could not process radar image for this timestamp.")

            random_delay = random.uniform(0.5, 2.0)
            print(f"  Waiting for {random_delay:.2f} seconds...")
            time.sleep(random_delay)

            current_dt -= timedelta(minutes=5)
            
    finally:
        print("\nClosing browser...")
        driver.quit()
        print("Crawler finished.")

if __name__ == "__main__":
    main()