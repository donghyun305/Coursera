import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd

path = 'baseball.db'
con = sq3.Connection(path)
query = """
SELECT *
    FROM allstarfull
    ;
"""
allstar_observations = pd.read_sql(query, con)
print(allstar_observations)

all_tables = pd.read_sql('SELECT * FROM sqlite_master', con)
print(all_tables)

best_query = """
SELECT playerID, sum(GP) AS num_games_played, AVG(startingPos) AS avg_starting_position
FROM allstarfull
GROUP BY playerID
ORDER BY num_games_played DESC, avg_starting_position ASC
LIMIT 3
"""
best = pd.read_sql(best_query, con)
print(best.head())
