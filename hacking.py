import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_images(url, folder_name='downloaded_images'):
    # Create a folder to save images
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create a session
    session = requests.Session()

    # Set headers to mimic a browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Get the content of the URL
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the URL: {e}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <li> tags with the specific class
    li_tags = soup.find_all('li', class_='thumbwook')

    # Download images from each <li> tag
    for li in li_tags:
        img = li.find('img')
        img_url = img.get('data-src')  # Get the data-src attribute
        if not img_url:  # Skip if no data-src found
            continue
        # Join the URL to handle relative paths
        img_url = urljoin(url, img_url)
        try:
            img_response = session.get(img_url, headers=headers)
            img_response.raise_for_status()  # Raise an error for bad responses
            # Get the image filename
            img_name = os.path.join(folder_name, os.path.basename(img_url))
            with open(img_name, 'wb') as f:
                f.write(img_response.content)
            print(f"Downloaded: {img_name}")
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

# Example usage
if __name__ == "__main__":
    url = input("Enter the URL to download images from: ")
    download_images(url)
