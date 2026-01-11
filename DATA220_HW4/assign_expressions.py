import sqlite3

class Expressions:
    sample_query = "SELECT * FROM conference_ranking WHERE rank = 'A*'"

    question1_1 = """
CREATE VIEW Universities_Pubs_Profile AS
SELECT
    a.affiliation AS university_name,
    COUNT(DISTINCT a.name) AS num_authors,
    SUM(p.count) AS total_pubs,
    SUM(p.count) / COUNT(DISTINCT a.name) AS avg_pubs,
    COUNT(DISTINCT CASE WHEN c.rank = 'A*' THEN a.name END) AS num_Astar_authors,
    SUM(CASE WHEN c.rank = 'A*' THEN p.count ELSE 0 END) AS total_Astar_pubs,
    SUM(CASE WHEN c.rank = 'A*' THEN p.count ELSE 0 END) / 
        COUNT(DISTINCT CASE WHEN c.rank = 'A*' THEN a.name END) AS avg_Astar_pubs,
    p.year
FROM
    author a
JOIN
    pub_info p ON p.name = a.name
LEFT JOIN
    conference_ranking c ON c.conf_abbr = p.conference
GROUP BY
    a.affiliation, p.year
"""

    question1_2 = """
CREATE VIEW Author_Profile AS
SELECT
    a.name AS author_name,
    p.year AS year,
    a.affiliation AS affiliated_university,
    SUM(p.count) AS num_papers,
    SUM(CASE 
        WHEN cr.rank = 'A*' THEN p.count
        ELSE 0 
    END) AS num_Astar_papers
FROM
    author a
JOIN
    pub_info p ON a.name = p.name
LEFT JOIN
    conference_ranking cr ON cr.conf_abbr = p.conference
GROUP BY
    a.name, p.year, a.affiliation
ORDER BY
    a.name, p.year
"""

    def question2(self, cursor, start_year, end_year, criteria, area, limit_val):
        query = """
SELECT
    p.name,
    COUNT(p.conference) AS author_rank
FROM
    pub_info AS p
JOIN
    field_conference AS f ON p.conference = f.conference
JOIN
    conference_ranking AS conf_rank ON f.conference = conf_rank.conf_abbr
WHERE
    p.year BETWEEN ? AND ?
    AND (? = 'all papers' OR conf_rank.rank = ?)
    AND (? = 'ALL' OR f.field = ?)
GROUP BY
    p.name
ORDER BY
    author_rank DESC,
    p.name DESC
LIMIT
    ?
"""
        params = (start_year, end_year, criteria, criteria, area, area, limit_val)
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results

    def question3(self, cursor, start_year, end_year, w_1, w_2, w_3, w_4, area, limit_val):
        query = """
WITH
    total_Astar_affiliation(affiliation, count_papers, count_authors) AS (
        SELECT
            a.affiliation,
            SUM(count),
            COUNT(DISTINCT pub_info.name)
        FROM
            pub_info
        NATURAL JOIN
            author AS a
        NATURAL JOIN
            field_conference
        JOIN
            conference_ranking ON field_conference.conference = conference_ranking.conf_abbr
        WHERE
            year BETWEEN ? AND ?
            AND rank = 'A*'
            AND (? = 'ALL' OR field = ?)
        GROUP BY
            a.affiliation
    ),
    total_affiliation(affiliation, count_papers, count_authors) AS (
        SELECT
            a.affiliation,
            SUM(count),
            COUNT(DISTINCT pub_info.name)
        FROM
            pub_info
        NATURAL JOIN
            author AS a
        NATURAL JOIN
            field_conference
        WHERE
            year BETWEEN ? AND ?
            AND (? = 'ALL' OR field = ?)
        GROUP BY
            a.affiliation
    )
SELECT
    a.affiliation,
    (? * (A_star.count_papers)) + 
    (? * (A_star.count_authors)) + 
    (? * (t.count_papers)) + 
    (? * (t.count_authors)) AS weighted_sum
FROM
    pub_info
NATURAL JOIN
    author AS a
JOIN
    total_Astar_affiliation AS A_star ON a.affiliation = A_star.affiliation
JOIN
    total_affiliation AS t ON a.affiliation = t.affiliation
WHERE
    year BETWEEN ? AND ?
GROUP BY
    a.affiliation
ORDER BY
    weighted_sum DESC
LIMIT
    ?
"""
        params = (start_year, end_year, area, area, start_year, end_year, area, area, 
                  w_1, w_2, w_3, w_4, start_year, end_year, limit_val)
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results

    question4_1 = """
CREATE TABLE IF NOT EXISTS Astar_Based_Rating(
    university_name TEXT PRIMARY KEY,
    score FLOAT DEFAULT 0.0
)
"""

    question4_2 = """
CREATE TABLE IF NOT EXISTS Balanced_Rating(
    university_name TEXT PRIMARY KEY,
    score FLOAT DEFAULT 0.0
)
"""

    question4_3 = """
CREATE TABLE IF NOT EXISTS General_Rating(
    university_name TEXT PRIMARY KEY,
    score FLOAT DEFAULT 0.0
)
"""

    question4_4 = """
INSERT INTO Astar_Based_Rating (university_name, score)
SELECT
    u.university_name,
    0.4 * (SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
    0.4 * (COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
    0.1 * (SUM(p.count)) +
    0.1 * COUNT(DISTINCT a.scholarid) AS score
FROM
    author a
JOIN
    pub_info p ON a.name = p.name
JOIN
    usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
    conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
    field_conference fc ON cr.conf_abbr = fc.conference
WHERE
    p.year BETWEEN 2014 AND 2024
GROUP BY
    u.university_name
ORDER BY
    score DESC
LIMIT
    50
"""

    question4_5 = """
INSERT INTO Balanced_Rating (university_name, score)
SELECT
    u.university_name,
    0.25 * (SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
    0.25 * (COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
    0.25 * (SUM(p.count)) +
    0.25 * COUNT(DISTINCT a.scholarid) AS score
FROM
    author a
JOIN
    pub_info p ON a.name = p.name
JOIN
    usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
    conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
    field_conference fc ON cr.conf_abbr = fc.conference
WHERE
    p.year BETWEEN 2014 AND 2024
GROUP BY
    u.university_name
ORDER BY
    score DESC
LIMIT
    50
"""

    question4_6 = """
INSERT INTO General_Rating (university_name, score)
SELECT
    u.university_name,
    0.1 * (SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
    0.1 * (COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
    0.4 * (SUM(p.count)) +
    0.4 * COUNT(DISTINCT a.scholarid) AS score
FROM
    author a
JOIN
    pub_info p ON a.name = p.name
JOIN
    usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
    conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
    field_conference fc ON cr.conf_abbr = fc.conference
WHERE
    p.year BETWEEN 2014 AND 2024
GROUP BY
    u.university_name
ORDER BY
    score DESC
LIMIT
    50
"""

    question5 = '''
CREATE TRIGGER Recalculate_Rankings_After_Insert
AFTER INSERT ON pub_info
WHEN NEW.year BETWEEN 2014 and 2024
BEGIN
INSERT OR REPLACE INTO Astar_Based_Rating (university_name, score)
SELECT
   u.university_name,
   0.4*(SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
   0.4*(COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
   0.1*(SUM(p.count)) +
   0.1*COUNT(DISTINCT a.scholarid) score
FROM
   author a
JOIN
   pub_info p ON a.name = p.name
JOIN
   usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
   conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
   field_conference fc ON cr.conf_abbr = fc.conference
WHERE
   p.year BETWEEN 2014 AND 2024
GROUP BY
   u.university_name     
ORDER BY
   score desc
limit 50;

INSERT OR REPLACE INTO Balanced_Rating (university_name, score)
SELECT
   u.university_name,
   .25*(SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
   .25*(COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
   .25*(SUM(p.count)) +
   .25*COUNT(DISTINCT a.scholarid) score
FROM
   author a
JOIN
   pub_info p ON a.name = p.name
JOIN
   usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
   conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
   field_conference fc ON cr.conf_abbr = fc.conference
WHERE
   p.year BETWEEN 2014 AND 2024
GROUP BY
   u.university_name     
ORDER BY
   score desc
limit 50;

INSERT OR REPLACE INTO General_Rating (university_name, score)
SELECT
   u.university_name,
   0.1*(SUM(p.count) FILTER (WHERE cr.rank = 'A*')) +
   0.1*(COUNT(DISTINCT a.scholarid) FILTER (WHERE cr.rank = 'A*')) +
   0.4*(SUM(p.count)) +
   0.4*COUNT(DISTINCT a.scholarid) score
FROM
   author a
JOIN
   pub_info p ON a.name = p.name
JOIN
   usnews_university_rankings u ON a.affiliation = u.university_name
JOIN
   conference_ranking cr ON p.conference = cr.conf_abbr
JOIN
   field_conference fc ON cr.conf_abbr = fc.conference
WHERE
   p.year BETWEEN 2014 AND 2024
GROUP BY
   u.university_name       
ORDER BY
   score desc
limit 50;

END;
'''
