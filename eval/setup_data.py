import json
import os

os.makedirs("eval/data/jds", exist_ok=True)
os.makedirs("eval/data/resumes", exist_ok=True)
os.makedirs("eval/results", exist_ok=True)

jds = {
    "01_backend_python.txt": """Job Title: Senior Backend Engineer (Python)
Requirements:
We are looking for a Backend Engineer with 3+ years of experience in Python, Flask, or Django.
Core skills required: PostgreSQL, Redis, Docker, Git, RESTful API design, and CI/CD pipelines.
Experience with Celery for background tasks and AWS (EC2, S3, RDS) is required.
Nice to have: Kubernetes, GraphQL, microservices architecture, and automated testing with pytest.
Must understand SQL query optimization, database indexing, and OOP principles.""",

    "02_frontend_react.txt": """Job Title: Senior Frontend Developer (React/TypeScript)
Responsibilities:
Build responsive web applications using React, TypeScript, Next.js, and Tailwind CSS.
State management with Redux Toolkit or Zustand.
Strong knowledge of HTML5, CSS3, ES6 JavaScript, responsive design, and CSS Grid/Flexbox.
Experience with REST APIs, GraphQL client integration, and unit testing using Jest and React Testing Library.
Familiarity with Webpack, Vite, Git, and npm/yarn packages.""",

    "03_fullstack_node.txt": """Job Title: Full-Stack Engineer (Node.js & React)
Requirements:
Proficiency with JavaScript and TypeScript across the full stack.
Backend: Node.js, Express, REST APIs, and GraphQL.
Databases: MongoDB, Mongoose, and PostgreSQL.
Frontend: React, Redux, HTML, CSS, and Tailwind CSS.
Cloud & DevOps: Docker, AWS (S3, Lambda), Git, GitHub Actions, and JWT authentication.""",

    "04_ai_ml_engineer.txt": """Job Title: AI / Machine Learning Engineer
Key Qualifications:
Strong proficiency in Python, PyTorch, TensorFlow, Scikit-learn, Pandas, and NumPy.
Experience in Natural Language Processing (NLP), text classification, embeddings, and Hugging Face transformers.
Hands-on experience with LLMs, prompt engineering, vector databases (Pinecone, ChromaDB, or FAISS).
Familiarity with Docker, FastAPI, Git, and ML deployment pipelines.
Background in machine learning algorithms, deep learning, computer vision, and linear algebra.""",

    "05_devops_cloud.txt": """Job Title: Cloud DevOps & Site Reliability Engineer
Required Skills:
Extensive experience with AWS (EC2, S3, IAM, VPC, EKS) and Google Cloud Platform (GCP).
Infrastructure as Code using Terraform and Ansible.
Container orchestration with Docker and Kubernetes.
CI/CD pipeline implementation using Jenkins, GitLab CI, or GitHub Actions.
Monitoring and alerting with Prometheus, Grafana, and ELK stack.
Scripting in Bash and Python, Linux system administration, and Git.""",

    "06_data_engineer.txt": """Job Title: Data Engineer (Big Data & Pipelines)
Requirements:
Expert in SQL, Python, and data modeling (star schema, snowflake schema).
Experience with Apache Spark, PySpark, Apache Kafka, and Apache Airflow.
Cloud data warehousing: Snowflake, Google BigQuery, or Amazon Redshift.
Experience with PostgreSQL, MongoDB, Docker, Git, and Linux.
Understanding of ETL / ELT pipeline design, data streaming, and REST APIs.""",

    "07_mobile_flutter.txt": """Job Title: Mobile Application Developer (Flutter / Dart)
Qualifications:
Proven experience building cross-platform mobile apps for iOS and Android using Flutter and Dart.
State management experience with Provider, Bloc, or Riverpod.
Integration with RESTful APIs, JSON parsing, WebSockets, and third-party SDKs.
Firebase integration: Cloud Firestore, Push Notifications, Firebase Auth.
Familiarity with Git, SQLite, mobile CI/CD, App Store and Google Play deployment.""",

    "08_backend_java.txt": """Job Title: Senior Java Backend Developer
Requirements:
5+ years of software development experience in Java (Java 11 or 17).
Strong experience with Spring Boot, Spring MVC, Spring Data JPA, and Hibernate.
Database experience with MySQL, PostgreSQL, and Redis caching.
Messaging systems: Apache Kafka or RabbitMQ.
Experience building scalable Microservices, REST APIs, Docker containers, and Kubernetes.
Proficient in Maven, Gradle, Git, JUnit testing, and CI/CD pipelines.""",

    "09_data_scientist.txt": """Job Title: Data Scientist / Statistical Modeler
Key Qualifications:
Proficiency in Python and R for data analysis, statistical modeling, and hypothesis testing.
Strong knowledge of SQL, Pandas, NumPy, Scikit-learn, and Statsmodels.
Data visualization with Matplotlib, Seaborn, Tableau, or Power BI.
Experience with machine learning techniques: regression, classification, clustering, random forest, and XGBoost.
Experience working with Jupyter notebooks, Git, and cloud platforms (AWS or Azure).""",

    "10_embedded_systems.txt": """Job Title: Embedded Systems & Firmware Engineer
Qualifications:
Strong expertise in C and C++ for embedded firmware development.
Hands-on experience with microcontroller architectures: ARM Cortex-M, STM32, or ESP32.
Understanding of communication protocols: UART, SPI, I2C, CAN, and Bluetooth.
Experience with RTOS (FreeRTOS) and Linux kernel / device drivers.
Proficient with Git, debugging tools (JTAG, GDB, logic analyzers), and hardware schematics."""
}

resumes = {
    "resume_01_backend.txt": """CANDIDATE 1 - BACKEND DEVELOPER
Summary: Software Engineer with 2 years of experience specializing in backend web development.
Technical Skills:
Languages: Python, JavaScript, SQL, HTML, CSS, Bash
Frameworks & Libraries: Flask, Django, FastAPI, Celery, SQLAlchemy
Databases: PostgreSQL, MySQL, Redis, SQLite
Tools & Cloud: Docker, Git, GitHub, AWS (EC2, S3), Linux, Postman, pytest
Experience:
- Built scalable RESTful APIs using Python, Flask, and PostgreSQL.
- Implemented asynchronous task processing with Celery and Redis.
- Containerized microservices using Docker and deployed applications on AWS EC2.
- Wrote automated unit and integration tests using pytest.""",

    "resume_02_fullstack.txt": """CANDIDATE 2 - FULL-STACK DEVELOPER
Summary: Full-Stack Developer with hands-on experience building modern, responsive web applications.
Technical Skills:
Frontend: React, TypeScript, Next.js, Redux, JavaScript, HTML5, CSS3, Tailwind CSS
Backend: Node.js, Express, REST APIs, GraphQL
Databases: MongoDB, PostgreSQL, Mongoose, Firebase
Tools: Git, GitHub, Docker, Vite, Webpack, Postman, Jest, npm
Experience:
- Developed single-page applications with React, TypeScript, and Tailwind CSS.
- Designed secure backend microservices using Node.js, Express, and JWT authentication.
- Managed NoSQL databases with MongoDB and relational data with PostgreSQL.
- Built reusable UI components and tested them using Jest and React Testing Library.""",

    "resume_03_data_ml.txt": """CANDIDATE 3 - DATA SCIENCE & AI INTERN
Summary: Data Science enthusiast with a strong foundation in machine learning, statistics, and NLP.
Technical Skills:
Languages: Python, R, SQL, C++
Data Science & ML: Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, SciPy
NLP & AI: Hugging Face transformers, NLTK, spaCy, text classification, embeddings, LLMs
Visualization & Tools: Matplotlib, Seaborn, Tableau, Jupyter Notebook, Git, Docker, Linux
Experience:
- Developed machine learning predictive models using Scikit-learn, XGBoost, and Pandas.
- Implemented NLP sentiment analysis and text classification pipelines using spaCy and PyTorch.
- Conducted exploratory data analysis and visualized insights using Matplotlib and Seaborn.
- Collaborated using Git and tracked experiments in Jupyter notebooks."""
}

for filename, content in jds.items():
    with open(os.path.join("eval/data/jds", filename), "w", encoding="utf-8") as f:
        f.write(content.strip())

for filename, content in resumes.items():
    with open(os.path.join("eval/data/resumes", filename), "w", encoding="utf-8") as f:
        f.write(content.strip())

# Ground-truth human-annotated labels (canonical, lowercased set of skills for each file)
labels = {
    "01_backend_python.txt": [
        "python", "flask", "django", "postgresql", "redis", "docker", "git", "rest",
        "ci/cd", "celery", "aws", "ec2", "s3", "rds", "kubernetes", "graphql", "microservices", "pytest", "sql"
    ],
    "02_frontend_react.txt": [
        "react", "typescript", "next.js", "tailwind", "redux", "zustand", "html", "css",
        "javascript", "rest", "graphql", "jest", "webpack", "vite", "git", "npm", "yarn"
    ],
    "03_fullstack_node.txt": [
        "javascript", "typescript", "node.js", "express", "rest", "graphql", "mongodb",
        "mongoose", "postgresql", "react", "redux", "html", "css", "tailwind", "docker", "aws", "s3", "lambda", "git", "github", "jwt"
    ],
    "04_ai_ml_engineer.txt": [
        "python", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "nlp",
        "hugging face", "transformers", "llm", "docker", "fastapi", "git", "machine learning", "deep learning", "computer vision"
    ],
    "05_devops_cloud.txt": [
        "aws", "ec2", "s3", "gcp", "terraform", "ansible", "docker", "kubernetes",
        "jenkins", "gitlab", "github", "ci/cd", "prometheus", "grafana", "elk", "bash", "python", "linux", "git"
    ],
    "06_data_engineer.txt": [
        "sql", "python", "spark", "pyspark", "kafka", "airflow", "snowflake", "bigquery",
        "redshift", "postgresql", "mongodb", "docker", "git", "linux", "etl", "rest"
    ],
    "07_mobile_flutter.txt": [
        "flutter", "dart", "ios", "android", "rest", "websockets", "firebase", "git", "sqlite", "ci/cd"
    ],
    "08_backend_java.txt": [
        "java", "spring", "hibernate", "mysql", "postgresql", "redis", "kafka",
        "rabbitmq", "microservices", "rest", "docker", "kubernetes", "maven", "gradle", "git", "junit", "ci/cd"
    ],
    "09_data_scientist.txt": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
        "tableau", "power bi", "machine learning", "git", "aws", "azure"
    ],
    "10_embedded_systems.txt": [
        "c", "c++", "arm", "rtos", "linux", "git", "embedded systems"
    ],
    "resume_01_backend.txt": [
        "python", "javascript", "sql", "html", "css", "bash", "flask", "django",
        "fastapi", "celery", "postgresql", "mysql", "redis", "sqlite", "docker", "git", "github", "aws", "ec2", "s3", "linux", "postman", "pytest", "rest", "microservices"
    ],
    "resume_02_fullstack.txt": [
        "react", "typescript", "next.js", "redux", "javascript", "html", "css", "tailwind",
        "node.js", "express", "rest", "graphql", "mongodb", "postgresql", "mongoose", "firebase", "git", "github", "docker", "vite", "webpack", "postman", "jest", "npm", "jwt", "microservices"
    ],
    "resume_03_data_ml.txt": [
        "python", "r", "sql", "c++", "pandas", "numpy", "scikit-learn", "pytorch",
        "tensorflow", "nlp", "transformers", "nltk", "spacy", "embeddings", "llm", "matplotlib", "seaborn", "tableau", "git", "docker", "linux", "machine learning"
    ]
}

with open("eval/labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

print("Created 10 JDs, 3 Resumes, and labels.json successfully!")

