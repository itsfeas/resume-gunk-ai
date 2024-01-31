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

[NAME] Raquib Lavani

[CONTACT] 587-598-3200 | r.raquib01@gmail.com | raquibkhan.com | linkedin.com/in/lavani | github.com/raquibk

[HEADER] Education

[SUBHEADER] University of Alberta Edmonton, AB
[SUBHEADER] Bachelor of Science in Software Engineering Co-op
[SUBHEADER] Sep. 2019 – May 2024

[NULL] • Technical Skills

[NULL] Languages: TypeScript, JavaScript, Rust, Python, C++, Java, Bash
[NULL] Frameworks: Next.js, React, Node, Flask, Jest, JUnit, Robotium, Selenium
[NULL] Developer Tools: AWS Fargate, AWS ECS, Docker, Git, Prisma, Figma, Android Studio

[HEADER] Experience

[SUBHEADER] Software Developer Intern
[SUBHEADER] InsideDesk Inc Toronto, ON
[SUBHEADER] May 2023 – December 2023

[NULL] • Secured a $27,000 contract by developing four robust web automation micro-services using Puppeteer
[NULL] • Led the overhaul of 24 micro-services by migrating to TypeScript and upgrading Docker images
[NULL] • Boosted performance of an in-house library by implementing in-memory caching for frequently requested resources
[NULL] • Implemented Bitbucket Pipeline configurations to build and upload Docker images to 4 development environments

[SUBHEADER] Software Engineer Intern
[SUBHEADER] Scotiabank Toronto, ON
[SUBHEADER] May 2022 – Aug. 2022

[NULL] • Developed and automated a multi-tiered compliance tool using VBA saving 35+ hours of weekly manual work
[NULL] • Devised a Python program to concatenate and format multiple csv files improving upload speed by 90%
[NULL] • Led the development of an automated dashboard by using ActiveX and ADO connections in VBA
[NULL] • Created a tool for Excel to SQL migration employed by the entire department improving lookup speed by over 400%

[SUBHEADER] Data Science Intern
[SUBHEADER] Scotiabank Toronto, ON
[SUBHEADER] Jan. 2022 – Apr. 2022

[NULL] • Designed an isolation forest fraud detection algorithm using scikit in Python improving fraud flagging by 12%
[NULL] • Analyzed 3 million+ data points using NumPy to compare fraud volume and rate in two different login streams
[NULL] • Automated a data pipeline using Demisto to push a scalable fraud detection tool to production
[NULL] • Improved data pull speed by 300% by utilizing multi-threading in Python to pull login data from IBM Security QRadar

[HEADER] Projects

[SUBHEADER] FoodBook | Android Studio, Java, Firebase, JUnit, Robotium, YAML

[NULL] • A fully functional android application for meal-planning and grocery shopping
[NULL] • Devised a comparison algorithm to detect missing ingredients and integrated a Firebase real-time database

[SUBHEADER] Inclusify | React, Node.js, SQLite, Javascript, Azure Computer Vision API, Hootsuite API

[NULL] • A Hack the North award-winning full-stack web application aimed to make social media posts inclusive
[NULL] • Implemented 4 parallel API calls to post on Twitter, caption images, and check for language errors

[SUBHEADER] ProtonNews | Next.js, Bun.js, FastAPI, Prisma, Cockroach DB, Cohere

[NULL] • A full-stack news website which uses sentiment analysis and summarizing to exclusively display positive news
[NULL] • Implemented a responsive front-end using server-side rendering and interfacing to CockroachDB using Prisma

[SUBHEADER] NLP Undergraduate Research | Python, Flask, Bash, PyTorch, kenlm

[NULL] • Research project under Dr. Carrie Demmans Epp lab to identify negative language transfer in translations
[NULL] • Implemented an n-gram language model and an RNN using syntactic data and ran performance benchmarks

[HEADER] Leadership Experience

[SUBHEADER] Undergraduate Teaching Assistant | Faculty of Computing Science

[NULL] • Led seminars, office hours, and grading for two computer courses, supporting 350+ students per semester

[SUBHEADER] Vice President Administration | Computing Science Club

[NULL] • Leading merchandise sales for 1000+ students, planning and execution of a graduation gala for 300+ students

[SUBHEADER] Director of Communications | Bhangra Dance Club

[NULL] • Founding member instrumental in obtaining $6000 in grants and organizing Diwali event for 160+ students


"""

result_list = split_lines_with_labels(input_text)

# Print the result
for item in result_list:
    print(item)
