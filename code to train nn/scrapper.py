import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# A Lock ensures that multiple threads don't write to data.txt at the exact same millisecond,
# which would corrupt the file structure.
file_write_lock = threading.Lock()

def fetch_arxiv_page(search_query, start_index, max_results=100):
    """
    Handles the network network hit safely and sequentially.
    Returns the raw XML text data if successful.
    """
    base_url = 'http://export.arxiv.org/api/query?'
    query_params = {
        'search_query': search_query,
        'start': start_index,
        'max_results': max_results
    }
    url = base_url + urllib.parse.urlencode(query_params)
    
    headers = {
        'User-Agent': 'AI-Training-Dataset-Builder/2.0 (mailto:your-email@example.com)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 503:
            print("⚠️ Server busy (HTTP 503). Backing off...")
            return "RETRY"
        else:
            print(f"❌ Failed connection: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Network error: {e}")
        return None

def process_and_save_chunk(xml_data, filename="data.txt"):
    """
    CPU-bound parsing and Disk-bound writing. 
    This runs entirely on background worker threads while the main thread fetches data.
    """
    if not xml_data or xml_data == "RETRY":
        return 0
    
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return 0
        
    entries = root.findall('atom:entry', namespaces)
    if not entries:
        return 0

    local_buffer = []
    
    # Process text layout inside memory (Thread-safe processing)
    for entry in entries:
        title_node = entry.find('atom:title', namespaces)
        summary_node = entry.find('atom:summary', namespaces)
        
        if title_node is None or summary_node is None:
            continue
            
        title = " ".join(title_node.text.strip().split())
        summary = " ".join(summary_node.text.strip().split())
        
        # Build block
        local_buffer.append(f"TITLE: {title}\nCONTENT:\n{summary}\n{'-'*40}\n\n")
            
    # Safely lock the disk write operation so threads don't override each other
    if local_buffer:
        with file_write_lock:
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(local_buffer)
                
    return len(local_buffer)

def scrape_topic(topic, total_pages=25, results_per_page=100, output_file="data.txt"):
    print(f"\n🚀 Launching pipeline for: {topic.upper()}")
    
    # Initialize our Thread Pool Executor for parsing and file operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        page = 0
        while page < total_pages:
            current_start = page * results_per_page
            print(f"📡 Requesting page {page+1}/{total_pages} (Offset: {current_start})...")
            
            raw_xml = fetch_arxiv_page(topic, current_start, results_per_page)
            
            if raw_xml == "RETRY":
                time.sleep(10)  # Extended backoff if throttled
                continue
            
            if not raw_xml:
                print("Skipping page due to download failure.")
                page += 1
                continue
                
            # TRICK: Submit parsing to background thread instantly. 
            # The main thread doesn't wait around for string cleanup or file writing!
            future = executor.submit(process_and_save_chunk, raw_xml, output_file)
            
            # Print feedback when background thread catches up
            def callback(f):
                count = f.result()
                if count > 0:
                    print(f"💾 Background Thread saved {count} articles to disk.")
            future.add_done_callback(callback)

            page += 1
            
            # MANDATORY: 3-second cooling gap on the main network thread 
            # to remain compliant with open-source endpoints.
            if page < total_pages:
                time.sleep(3.2)

def main():
    topics = [
        "english literature",
        "english language learning",
        "english grammar",
        "english vocabulary",
        "english comprehension",
        "english conversation"
    ]
    
    output_file = "data.txt"
    TOTAL_PAGES = 20
    ARTICLES_PER_PAGE = 100
    
    if os.path.exists(output_file):
        print(f"Found existing data.txt. Appending...")
    else:
        print(f"Generating clear output file at: {output_file}")

    for topic in topics:
        scrape_topic(topic, total_pages=TOTAL_PAGES, results_per_page=ARTICLES_PER_PAGE, output_file=output_file)

    print(f"\n🎉 Finished! Processed up to {TOTAL_PAGES * ARTICLES_PER_PAGE} articles per topic safely.")

if __name__ == "__main__":
    main()
    
    