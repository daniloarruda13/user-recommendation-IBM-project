# user-recommendation-IBM-project

This project contains a recommendation engine using the IBM-Watson data. This IBM platform is a website that provides articles about programming, data science, among others.

## Files and Data

-   **Recommendations_with_IBM.html / Recommendations_with_IBM.ipynb** : Those are the notebooks containing the Python script to perform the recommendations.

-   **user_item_matrix.p** : This is the user-item matrix in pickly format. This was used throughout the notebook.

-   **project_tests.py, top_5.p, top_10.p, top_20.p** : Those are files provided by Udacity to test the functions in the notebook.

-   ***DATA*****:**

    -   **articles_community.csv** : This is a spreadsheet with the descriptions of the articles, containing descriptions and titles.

    -   **user-item-interactions.csv** : This is a spreadsheet with the users_id and their interaction (articles they read) to the articles.

## Installation

-   Python 3.x

-   [Nltk](https://www.nltk.org/)

-   [Pandas](http://pandas.pydata.org/)

-   [Numpy](https://numpy.org/)

-   [scikit-learn](http://scikit-learn.org/stable/)

## Overview

The notebook includes a few recommendation strategies:

*Rank-based recommendations*: Items are ranked based on popularity or user ratings, and the top-ranked ones are recommended.

*Collaborative recommendations*: Recommendations are made by analyzing user behavior and similarities among users.

*Content-based recommendations*: Items are recommended based on their attributes and matching them with the user's preferences.

*Matrix factorization (SVD)*: A technique that decomposes a user-item matrix to predict missing ratings or recommend new items.
