import sqlite3
from relational_algebra import *

class Expressions:
    # You can manually define expressions here or load from the uploaded file
    # Example Query: Find Databases conference and their rankings
    sample_query = Projection(
            ThetaJoin(
                Selection(
                    Relation("field_conference"),
                    Equals("field", "Databases")
                ),
                Relation("conference_ranking"),
                Equals("field_conference.conference", "conference_ranking.conf_abbr")
            ),
            ["conference_ranking.conf_abbr", "conference_ranking.rank"]
        )

    # Uncomment and define other expressions
    expression1 = Projection(
        ThetaJoin(
            NaturalJoin(
                Selection(
                    Relation("author"),
                    Equals("affiliation", "University of California, Berkeley")
                ),
                Selection(
                    Relation("pub_info"),
                    And(
                        GreaterEquals("year", 2010),
                        LessEquals("year", 2024)
                    )
                )
            ),
            
            Projection(
                Selection(
                    Relation("field_conference"),
                    Equals("field", "Databases")
                ),
                ["conference"]
            ),
            Equals("pub_info.conference", "field_conference.conference")
        ),
    ["author.name", "pub_info.conference", "pub_info.year", "pub_info.count"]
    )

    expression2 = Projection(
        Selection(Relation("field_conference"),Equals("major", "Computer Science")),
	    ["conference"]
    ) - Projection (
        NaturalJoin(
                NaturalJoin(
                        Selection(Relation("author"), Equals("affiliation", "University of California, Irvine") 
                                  ), 
                        Relation("pub_info")
                ), 
                Selection(Relation("field_conference"),Equals("major", "Computer Science") 
                          )
        ),
        ["pub_info.conference"]
    )
    
    pub_info_db = Projection(
                        ThetaJoin( 
                                Selection(Relation("field_conference"), Equals("field", "Databases")),
                                Relation("pub_info"),
                                Equals("field_conference.conference", "pub_info.conference")
                        ),
        ["pub_info.name", "pub_info.conference", "pub_info.year", "pub_info.count"]
    )

    min_conf = Projection(
        ThetaJoin(
            Rename(pub_info_db, "pub_info_db1"), 
            Rename(pub_info_db, "pub_info_db2"), 
            And(Equals("pub_info_db1.year", "pub_info_db2.year"), 
                And(Equals("pub_info_db1.conference", "pub_info_db2.conference"),
                    LessThan("pub_info_db1.count", "pub_info_db2.count"))) 
        ), 
         ["pub_info_db1.name", "pub_info_db1.conference", "pub_info_db1.year", "pub_info_db1.count"]
    )
    max_conf = pub_info_db - min_conf

    expression3 = Projection(
        ThetaJoin(
            Rename(max_conf, "max_conf"), 
            Relation("author"),
            Equals("max_conf.name", "author.name")
        ),
        ["max_conf.conference","max_conf.year","max_conf.name", "author.affiliation"]
    )
    db_conf = Projection(
            Selection(Relation("field_conference"), Equals("field", "Databases")),
            ["conference"]
    )
    pub_2020_2024 = Selection(
                Relation("pub_info"),
                And(GreaterEquals("year", 2020), LessEquals("year", 2024))
    )

    db_pub_authors = Projection(
        NaturalJoin(
                Relation("author"),
                Rename(pub_2020_2024, "pub_2020_2024")
        ),
        ["author.affiliation", "pub_2020_2024.conference"]
    )
    univ_db_conf = Projection(
            NaturalJoin(
            Rename(db_conf, "db_conf"), 
            Rename(db_pub_authors, "db_pub_authors")
        ),
        ["db_pub_authors.affiliation", "db_conf.conference"]
    )

    expression4 = Division(
            Rename(univ_db_conf, "univ_db_conf"), 
            Rename(db_conf, "db_conf")
    )

    uci_rank = Projection(
        Selection(
            Relation("usnews_university_rankings"),
            Equals("university_name", "University of California, Irvine")
        ),
        ["university_name","rank"]
    )
    
    uni_rank_higher_uci = Projection(
        ThetaJoin(
            Relation("usnews_university_rankings"),
            Rename(uci_rank, "uci_rank"),
            And(LessThan("usnews_university_rankings.rank", "uci_rank.rank"), 
                Not(Equals("usnews_unversity_rankings.rank", "uci_rank.rank")))
        ),
        ["usnews_university_rankings.university_name"]
    )
    uni_rank_higher_uci = Rename(uni_rank_higher_uci, 
                                 mapping={"usnews_university_rankings.university_name": "affiliation"})

    a_star_conferences = Projection(
        ThetaJoin(
            Selection(Relation("conference_ranking"), Equals("rank", "A*")),
            Selection(Relation("field_conference"), Equals("field", "Databases")), 
            Equals("conf_abbr", "conference")
        ),
        ["conference"]
    )

    a_star_publications = Projection(
        NaturalJoin(
            Relation("pub_info"),
            Rename(a_star_conferences, "a_star_conferences")
        ),
        ["pub_info.name", "conference"]
    )

    univ_a_star_publications = Projection(
        NaturalJoin(
            Rename(a_star_publications, "a_star_publications"),
            Relation("author")
        ),
        ["author.affiliation"]
    )
    
    universities_with_no_a_star = Rename(uni_rank_higher_uci,"uni_rank_higher_uci") - Rename(univ_a_star_publications, "univ_a_star_publications")
    
    expression5 = Projection(
        Rename(universities_with_no_a_star, "universities_with_no_a_star"),
        ["affiliation"]

    )

    #BONUS QUESTION
    #First we would use this data to compare universities by comparing their university rankings for prospective students. 
    #We would do a self join to find universities with the best rankings. 
    #To compare universities specific major, we would have to look at the faculty. Students and recent graduates will look for a strong faculty
    #We would look at the count of faculty in a specific field whether they published in renowned conferences.
    #Prospective students will also look for info on acceptance rate, cost/ tuition, and enrollment which where we will use University ranking
    #Recent graduates looking for employment will look for funding for their interested research field
    #Thus we will need to join the author table, publication info table, and field to conference and group by university 
    # We would display the faculty of each university, perstige of faculty (awards & acm fellow), count of publication records in conferences in a specific field 
    # We could use this data to also find partnerships between universities within a research field but this could only be done if the paper title is given to link many authors to that title 

    # Example query execution 
    #sql_con = sqlite3.connect('/Users/ingridamerica/Downloads/sample220P.db')  # Ensure the uploaded database is loaded here
    #result =  expression5.evaluate(sql_con=sql_con)
    #print(result.rows)