def split_lines_with_labels(text):
    lines = text.split('\n')
    labeled_list = [] 

    for line in lines:
        if line.strip():  # Skip empty lines
            words = line.split()
            label = words[0] [1:-1] 
            labeled_list.extend([(label, word) for word in words[1:] ] )

    return labeled_list

# Example usage with the provided text
input_text = """
[NAME] Alishbah Farooq
[CONTACT] (587)-991-9988 alishbah@ualberta.ca www.linkedin.com/in/alishbah

[HEADER] EDUCATION
[SUBHEADER] BSc. Specialization in Mathematics and Finance
[SUBHEADER] University of Alberta
[HEADER] PROJECTS

[SUBHEADER] 2nd Place in FemTech Futures Hackathon

[SUBHEADER] | April 2023

[NULL] Using Unity, worked in a team to make a maze including obstacles and a car to navigate through

[NULL] Meta Tic Tac Toe game incorporating traditional and numerical Tic Tac Toe in Python
[NULL] Spite and Malice card game between two players in Python

[HEADER] SKILLS

[NULL] Python, C, SQLite
[NULL] Native in Urdu
[NULL] Fluent in English and French

[HEADER] EXPERIENCE
[SUBHEADER] | November 2021 - PresentStudent Contractor
[SUBHEADER] Alberta Machine Intelligence Institute (Amii)

[NULL] Support Machine Learning Scientists at Amii coach startup companies in adopting ML in their SCALE AI -
[NULL] Supply Chain AI West Program
[NULL] Work within a team of 5 individuals
[NULL] Provide insight around various ML topics and processes

[SUBHEADER] VP Finance
[SUBHEADER] Ada's Team

[SUBHEADER] | May 2021 - Present

[NULL] Created and managed the budget of $40 000 for the fiscal year
[NULL] Tracked all incoming and outgoing money flow in Excel sheet
[NULL] Worked within team of 5-8 to organize over 6 technology-related workshops for university students

[SUBHEADER] Summer Day Camp Counsellor | Full-time | July - August 2019
[SUBHEADER] YMCA

[NULL] Had full time responsibility for safety of up to 20 children between ages 3-12
[NULL] Communicated between about 10 camps and 20 counsellors to ensure safety of all registered campers
[NULL] Was responsible for medical and personal records of up to 20 children

[HEADER] VOLUNTEER/EXTRACURRICULARS

[SUBHEADER] NeurAlberta Tech Machine Learning Workshops                                            | Present
[SUBHEADER] WISER - Data Science 101: Intro to Tableau Workshop                                   | May 2021
[SUBHEADER] Mentor in Ada's Mentors ~ 7.5h                                                                           |  November 2020 - April 2020
[SUBHEADER] Grace Hopper Celebration Attendee                                                                  |  September 2020, 2021
[SUBHEADER] Student Devcon Attendee                                                                                    |   January 2021
[SUBHEADER] Wisest Choices Conference Volunteer  ~4h                                                      |  February 2020

[HEADER] ACHIEVEMENTS & CERTIFICATIONS

[SUBHEADER] Certified bilingual with B2 DELF French Diploma | March 2019
[SUBHEADER] Distinguished Service Award | May 2019

"""

result_list = split_lines_with_labels(input_text)

# Print the result
for item in result_list:
    print(item[0], item[1])
