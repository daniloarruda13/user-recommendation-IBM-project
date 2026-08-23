# IBM article recommendation engine

This repository explores recommendation strategies for implicit article-reading
data from the former IBM Watson Studio platform. The original Udacity-style
notebook covers popularity, user-user collaborative filtering, text clustering,
and dense singular value decomposition (SVD).

The reusable `ibm_recommender` package provides deterministic data preparation,
popularity and collaborative recommendations, explicit cold-start behavior, and
ranking metrics suitable for implicit feedback.

## Data

The committed CSV files are small enough for local analysis:

- `data/user-item-interactions.csv`: 45,993 interaction events, 714 interacted
  articles, 5,148 identified users, and 17 anonymous events.
- `data/articles_community.csv`: 1,056 rows representing 1,051 unique articles.

Article IDs appear as both values such as `1429.0` and integers across the two
sources. The package converts every integer-like ID to a canonical Python integer
before joining, ranking, or checking whether an item has already been seen.

Anonymous activity remains valid for global popularity. For personalization,
`encode_users(..., anonymous="drop")` excludes those rows by default instead of
treating 17 unrelated anonymous events as one person. An explicit
`anonymous="single"` mode is available when that assumption is intended.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

To inspect the historical notebook and its content-clustering sections, install
the optional notebook stack:

```bash
python -m pip install -r requirements.txt
```

## Usage

```python
from ibm_recommender import (
    article_names,
    create_user_item_matrix,
    encode_users,
    load_interactions,
    recommend_article_ids,
    top_articles,
)

interactions = load_interactions("data/user-item-interactions.csv")

# Popularity includes anonymous events.
print(top_articles(interactions, 10))

# Personalized interactions exclude anonymous rows.
personalized = encode_users(interactions)
user_item = create_user_item_matrix(personalized)
ids = recommend_article_ids(20, personalized, 10, user_item=user_item)
print(article_names(ids, interactions))
```

Known users receive articles scored by similarity-weighted neighbor activity,
with global popularity and article ID used as deterministic tie-breakers. Seen
articles are always excluded. Unknown users and users without positive neighbors
receive unseen popularity recommendations.

## Evaluation

The data records reads, not ratings. A missing user-item entry means “not
observed,” not “disliked.” Consequently, accuracy across a dense binary matrix is
dominated by zeros and is not an informative recommender metric.

The package supplies `precision_at_k`, `recall_at_k`, and `hit_rate_at_k`. For an
offline evaluation, hold out each eligible user’s later interactions, generate
recommendations from earlier events, and calculate ranking metrics only for users
and items the method can score. Report cold-start coverage separately. In
production, an A/B test should assess actual reading or engagement outcomes.

## Historical notebook caveats

`Recommendations_with_IBM.ipynb` and the rendered HTML remain historical course
artifacts. The package addresses several issues without rewriting cached outputs:

- `get_top_sorted_users` performs two independent sorts, making interaction count
  the effective primary key instead of similarity.
- Several collaborative ties are arbitrary; the package uses explicit stable
  similarity, activity, and ID ordering.
- The content recommender mixes string and integer article IDs when excluding
  seen items and can stop a cluster scan prematurely.
- The dense SVD evaluation predicts mostly unobserved zeros and reports accuracy,
  which can appear high without producing useful top-N recommendations.
- The original email mapper groups missing emails as a user. The package makes
  the anonymous-user policy explicit.

The four `.p` files are historical Python pickle artifacts. Pickle can execute
code while loading and is version-sensitive; load these files only when their
source is trusted. New code should regenerate the user-item matrix from the CSV
rather than depend on the committed 29 MB pickle.

The notebook’s `project_tests.py` is a course answer checker, not an isolated
automated test suite: it reads repository data at import time and prints feedback
instead of using test assertions consistently.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover ID normalization, anonymous users, deterministic popularity, binary
matrices, neighbor tie-breaking, seen-item exclusion, cold starts, ranking
metrics, complete committed CSV dimensions, the real popularity baseline, and
the full 5,148-by-714 real user-item matrix.

## Repository structure

- `ibm_recommender/data.py`: loaders, ID normalization, and stable user encoding.
- `ibm_recommender/ranking.py`: popularity and ordered title lookup.
- `ibm_recommender/matrix.py`: binary matrices and neighbor ordering.
- `ibm_recommender/recommend.py`: collaborative and cold-start recommendations.
- `ibm_recommender/metrics.py`: top-k implicit-feedback metrics.
- `tests/`: synthetic and committed-data regression tests.
- `Recommendations_with_IBM.ipynb` / `.html`: historical exploratory report.
- `project_tests.py`: retained course answer checker.

## License and data reuse

No software or data license is currently declared. Copyright remains with the
respective authors and data providers; obtain permission before reuse beyond
applicable legal rights.
