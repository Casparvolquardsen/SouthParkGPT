import csv
import os

import requests
from bs4 import BeautifulSoup


def get_name(url):
    return url.split('/')[-1]


def extract_and_save_table(url, save_dir='southpark_scripts'):
    try:
        # Fetch HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the element with id="Script"
        # The table containing the script is always the next element after the script element in the DOM
        script_element = soup.find(id='Script')

        if not script_element:
            # If the script element is not found, try finding the element with id="Scripts"
            script_element = soup.find(id='Scripts')

        if script_element:
            # Find the table following the script element
            table = script_element.find_next('table')
            if table:
                # Extract data from the table and save to CSV file
                rows = table.find_all('tr')
                with open(
                    f'{save_dir}/{url.split("/")[-2]}.csv',
                    'w',
                    newline='',
                    encoding='utf-8',
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    for row in rows:
                        cols = row.find_all(['td', 'th'])
                        cols = [col.get_text(strip=True) for col in cols]
                        writer.writerow(cols)
                print(f"Table from {url} saved successfully.")
                return True
            else:
                print(f"No table found after the script element in {url}.")
        else:
            print(f"No element with id='Script' found in {url}.")
    except Exception as e:
        print(f"Error occurred while processing {url}: {e}")


# Fetch the main page of South Park scripts where all the seasons are listed
response = requests.get("https://southpark.fandom.com/wiki/Portal:Scripts")
response.raise_for_status()
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

# Find all <a> tags within the div with class="wikia-gallery-item"
a_tags = soup.select('div.wikia-gallery-item a')

# Extract href links (filter out .png links)
season_links = set(
    [
        "https://southpark.fandom.com" + a['href']
        for a in a_tags
        if a['href'] and not a['href'].endswith('.png')
    ]
)

season_episodes_map = dict()

# Extract episode links for each season
for season_link in season_links:
    episode_links = set()
    response = requests.get(season_link)
    response.raise_for_status()
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    a_tags = soup.select('div.wikia-gallery-item a')
    episode_links.update(
        [
            "https://southpark.fandom.com" + a['href']
            for a in a_tags
            if a['href'] and not a['href'].endswith('.png')
        ]
    )
    season_episodes_map[season_link.split('/')[-1]] = episode_links


episode_counter = 0
error_counter = 0

# Print extracted href links
for season_link in season_links:
    season_name = season_link.split('/')[-1]
    print(season_name)
    print(
        [
            episode_link.split('/')[-2]
            for episode_link in season_episodes_map[season_link.split('/')[-1]]
        ]
    )

    # create a directory for each season
    os.makedirs(f"southpark_scripts/{season_name}", exist_ok=True)

    for episode_link in season_episodes_map[season_name]:
        success = extract_and_save_table(episode_link, f"southpark_scripts/{season_name}")
        if success:
            episode_counter += 1
        else:
            error_counter += 1

print(f"Total episodes saved: {episode_counter}")
print(f"Total errors: {error_counter}")
