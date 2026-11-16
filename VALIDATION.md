# Member 1 validation

Validated on November 15, 2026 with Python 3.12.14 in a fresh virtual environment.
The pinned dependencies installed successfully and `pip check` reported no
broken requirements.

| Check | Observed result |
| --- | --- |
| Official MovieLens ZIP checksum | Matches cd245b17a1ae2cc31bb14903e1204af3 |
| Five raw CSV checksums | All matched the recorded SHA-256 hashes |
| SQLite integrity | Passed |
| Exported training rows | All 15,915,724 rows exactly matched the database |
| Exported test rows | All 4,046,913 rows exactly matched the database |
| User eligibility and split sizes | All 138,292 users had at least 20 linked ratings and exactly floor(0.8 × n) train rows |
| Chronological boundaries | Every user's latest training timestamp was less than or equal to their earliest test timestamp |
| Timestamp ties at the boundary | 34,272 users, resolved by ascending canonical movie ID |
| Shared catalog | 5,000 unique MovieLens IDs and 5,000 unique TMDB IDs, all with nonempty titles and overviews |
| Popularity counts | Every catalog count matched the count computed from training ratings |
| Model persistence | Reloaded sparse vectors and IDF weights matched a fresh fit |
| TF-IDF matrix | 5,000 movie rows and 182,361 text features |
| Recommendation export | All 50,000 rows matched predictions from the saved model |
| Ranking integrity | Ten distinct in-catalog recommendations per seed, with the seed excluded |
| Handoff integrity | Ten output files passed checksum verification |
| Regression suite | Three tests passed for alias handling, chronological split, train popularity, model round-trip, exclusions, and CSV exports |

The test suite runs with `python -m unittest discover -s tests -v`.
The delivered outputs can be checked with `python -m movie_rec.cli verify`.

The Toy Story query returned Toy Story 2 first, followed by Small Soldiers.
This confirms the normal recommendation entry point runs with the delivered
model. Ranking accuracy metrics belong to the shared evaluation in Member 2's assignment.
