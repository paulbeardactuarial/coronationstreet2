# %%
import requests
import pandas as pd
from bs4 import BeautifulSoup
import string
import re
from typing import List, Dict, Any
import time
import os


# %%
# Function to scrape fandom wiki for list of character names in Coronation St.
def scrape_corrie_chr_names(
        base_url_letters: str = "https://coronationstreet.fandom.com/wiki/Category:List_of_main_character_appearances?from="
):
    """
    Scrapes main character names from the Coronation Street Fandom wiki.

    This function iterates through each letter of the English alphabet, appends it
    to the base URL to generate a complete category page URL, and scrapes all character
    names listed under that page. The function collects and deduplicates the names
    before returning them.

    Args:
        base_url_letters (str): The base URL to which each alphabet letter is appended
                                to form the full URL for each character list page.
                                Defaults to the Coronation Street Fandom character appearance category.

    Returns:
        List[str]: A list of unique main character names scraped from the wiki.

    Dependencies:
        - requests
        - bs4 (BeautifulSoup)
        - re
        - string
    """

    character_names = []

    for letter in string.ascii_uppercase:
        char_list_url = f"{base_url_letters}{letter}"
        response = requests.get(char_list_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup.select(".category-page__member-link"):
            name = element.text
            name = re.sub(" - List of appearances", "", name)
            character_names.append(name)

    character_names = list(set(character_names))

    return character_names


# %%
# Function to scrape fandom wiki for a single character from the show
def scrape_fandom_wiki_single_chr(character, base_url):
    """
    Get dataframe containing information for a single character bio 

    Args: 
        character: string (the character from Corrie)
        base_url: string (the core url of the wiki)

    Returns: 
        pd.DataFrame object with the information for the Coronation Street character
    """

    char_text = character.replace(" ", "_")
    char_bio_url = f"{base_url}{char_text}"

    try:
        response = requests.get(char_bio_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Pull the key stats for the character
        char_stat = [element.text for element in soup.select(
            ".pi-secondary-font")]
        if char_stat:  # Remove first element
            char_stat = char_stat[1:]

        char_value = [element.text for element in soup.select(".pi-font")]

        # Create DataFrame for this character
        char_data = pd.DataFrame({
            'Field': char_stat,
            'Value': char_value
        })

        return char_data
    except Exception as e:
        print(f"Error scraping {character}: {e}")
        return pd.DataFrame(columns=['Field', 'Value'])

# %%


def scrape_corrie_data():
    """
    Scrape character data from Coronation Street wiki.

    Args:
        save_results: Boolean determining whether to save results to CSV file

    Returns:
        pandas DataFrame containing character data
    """

    all_char_names = scrape_corrie_chr_names()

    # Base URL code for scraping character data
    base_url_characters = "https://coronationstreet.fandom.com/wiki/"

    char_data_dict = {}

    char_data_dict = {character: scrape_fandom_wiki_single_chr(
        character, base_url_characters) for character in all_char_names}

    # Combine all DataFrames into one
    character_data = pd.concat(char_data_dict, names=[
                               'Character']).reset_index(level=0)

    return character_data


# %%
# Example usage
if __name__ == "__main__":
    os.makedirs("./Data", exist_ok=True)
    character_data = scrape_corrie_data()
    character_data.to_csv("./Data/character_data.csv", index=False)
    print(character_data.head())

# %%
