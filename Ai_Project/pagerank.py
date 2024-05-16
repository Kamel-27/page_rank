from importlib.metadata import distribution
import os
import random
import re
import sys
from unittest import result

# Constants for PageRank calculation
DAMPING = 0.85
SAMPLES = 10000

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    
    # Call the crawl function to parse the corpus directory and extract links
    corpus = crawl(sys.argv[1])
    
    # Calculate PageRank using sampling method
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    # Print the PageRank results from sampling
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    # Calculate PageRank using iterative method
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    # Print the PageRank results from iteration
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            # Use regular expression to find links in HTML content
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            # Remove self-referencing links and store the remaining links
            pages[filename] = set(links) - {filename}

    # Filter out links that do not point to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dist = {}
    links = corpus[page]

    # If the page has no outgoing links, choose randomly among all pages
    if len(links) == 0:
        for link in corpus:
            dist[link] = 1 / len(corpus)
    else:
        # Calculate probability distribution based on PageRank algorithm
        for link in corpus:
            dist[link] = (1 - damping_factor) / len(corpus)
        for link in links:
            dist[link] += damping_factor / len(links)
    
    return dist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_rank = {}
    for key in corpus.keys():
        page_rank[key] = 0
    # Choose a random page to start with
    sample_page = random.choice(list(corpus.keys()))
    # Sample pages according to the transition model
    for i in range(1, n):
        page_rank[sample_page] += 1
        page_dist = transition_model(corpus, sample_page, damping_factor)
        sample_page = random.choices(population=list(page_dist), weights=page_dist.values(), k=1)[0]

    # Normalize the PageRank values
    for key in page_rank.keys():
        page_rank[key] /= n

    return page_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    D = damping_factor
    random_choice = (1 - D) / N
    page_rank = {}
    calculated_pr = {}
    condition = True

    # Initialize each page's rank with 1/N
    for page in corpus:
        page_rank[page] = 1 / N

    # Iterate until convergence
    while condition:
        for current_page in page_rank:
            sum = 0.0
            for page in corpus:
                if current_page in corpus[page]:
                    NumLinks = len(corpus[page])
                    sum += page_rank[page] / NumLinks
                if not corpus[page]:
                    sum += page_rank[page] / N

            d_times_sum = D * sum
            calculated_pr[current_page] = random_choice + d_times_sum

        condition = False
        # Check for convergence
        for current_page in page_rank:
            if abs(page_rank[current_page] - calculated_pr[current_page]) > 0.001:
                condition = True
            page_rank[current_page] = calculated_pr[current_page]
    
    return page_rank

if __name__ == "__main__":
    main()
   


