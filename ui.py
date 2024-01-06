# Import necessary libraries and modules
from flask import Blueprint, render_template, request
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
from SPARQLWrapper import SPARQLWrapper, JSON
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# ontology namespace
_namespace = Namespace(" <http://www.semanticweb.org/margaret.gichuhi/ontologies/2024/0/CropYieldOntology/>")

# SPARQL endpoint
sparql_endpoint = "http://localhost:9999/blazegraph/sparql"

# SPARQL wrapper
sparql = SPARQLWrapper(sparql_endpoint)

# SPARQL query template
sparql_query_template = f"""
    PREFIX : {_namespace}
    SELECT DISTINCT ?crop ?country ?temperature ?rainfall ?pesticide ?yield ?year
    WHERE {{
        ?obs a :CropYieldObservation ;
             :hasCrop ?crop ;
             :hasCountry ?country ;
             :hasWeatherCondition ?wc ;
             :hasPesticideUsage ?pu ;
             :hasYieldInHGHA ?yield ;
             :hasYear ?year .
        ?wc :hasAverageTemperature ?temperature ;
            :hasAverageRainfall ?rainfall .
        ?pu :hasPesticideAmount ?pesticide .
    }}
    
    """

# UI blueprint
ui = Blueprint("ui", __name__)


# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return " ".join(tokens)


# NLU function for intent extraction and entity recognition
def process_nlu(user_query):
    doc = nlp(user_query)

    intent = None
    entities = []

    for token in doc:
        if token.ent_type_:
            entities.append({"text": token.text, "type": token.ent_type_})

    # Check for specific patterns in the tokenized text
    text_tokens = [token.text for token in doc]
    text_lower = ' '.join(text_tokens)

    if 'highest yield' in text_lower:
        intent = 'highest_yield'
    if 'lowest yield' in text_lower:
        intent = 'lowest_yield'
    elif 'temperature affect' in text_lower:
        intent = 'temperature_effect'
    elif 'pesticide' in text_lower:
        intent = 'pesticide_usage'
    elif 'production' in text_lower:
        intent = 'crop_production'
    elif 'crops' in text_lower:
        intent = 'crop_grown'

    return {"intent": intent, "entities": entities}


# NLQF function for translating NLU output into SPARQL query
def translate_to_sparql(nlu_output):
    intent = nlu_output["intent"]
    entities = nlu_output["entities"]

    if intent == "highest_yield":
        return generate_highest_yield_query(entities)
    elif intent == "lowest_yield":
        return generate_lowest_yield_query(entities)
    elif intent == "temperature_effect":
        return generate_temperature_effect_query(entities)
    elif intent == "pesticide_usage":
        return generate_pesticide_usage_query(entities)
    elif intent == "crop_production":
        return generate_crop_production_query(entities)
    elif intent == "crop_grown":
        return generate_crop_grown_query(entities)
    else:
        return sparql_query_template


def generate_highest_yield_query(entities):

    country_entity = next((entity["text"] for entity in entities if entity["type"] == "GPE"), None)
    year_entity = next((entity["text"] for entity in entities if entity["type"] == "DATE"), None)

    if country_entity and year_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT ?crop ?yield
            WHERE {{
                ?obs :hasCountry :{country_entity} ;
                     :hasYear <http://www.semanticweb.org/margaret.gichuhi/ontologies/2024/0/CropYieldOntology#{year_entity}> ;
                     :hasCrop ?crop ;
                     :hasYieldInHGHA ?yield .
            }}
            ORDER BY DESC(?yield)
        """
    else:
        return "INVALID_QUERY"


def generate_lowest_yield_query(entities):
    country_entity = next((entity["text"] for entity in entities if entity["type"] == "GPE"), None)
    year_entity = next((entity["text"] for entity in entities if entity["type"] == "DATE"), None)

    if country_entity and year_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT ?crop ?yield
            WHERE {{
                ?obs :hasCountry :{country_entity} ;
                     :hasYear <http://www.semanticweb.org/margaret.gichuhi/ontologies/2024/0/CropYieldOntology#{year_entity}> ;
                     :hasCrop ?crop ;
                     :hasYieldInHGHA ?yield .
            }}
            ORDER BY ASC(?yield)
            Limit 1
        """
    else:
        return "INVALID_QUERY"


def generate_temperature_effect_query(entities):
    crop_entity = next((entity["text"] for entity in entities if entity["type"] in ["PRODUCT", "ORG"]), None)
    country_entity = next((entity["text"] for entity in entities if entity["type"] == "GPE"), None)

    if crop_entity and country_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT ?year ?temperature ?yield
            WHERE {{
                ?obs :hasCountry :{country_entity} ;
                     :hasYear ?year ;
                     :hasCrop :{crop_entity} ;
                     :hasYieldInHGHA ?yield ;
                     :hasWeatherCondition ?wc .
                ?wc :hasAverageTemperature ?temperature .
            }}
            ORDER BY ?year
        """
    else:
        return "INVALID_QUERY"


def generate_pesticide_usage_query(entities):
    crop_entity = next((entity["text"] for entity in entities if entity["type"] in ["PRODUCT", "ORG"]), None)
    threshold_entity = next((entity["text"] for entity in entities if entity["type"] == "CARDINAL" and entity["text"].isdigit()), None)
    year_entity = next((entity["text"] for entity in entities if entity["type"] == "DATE"), None)

    if crop_entity and threshold_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT ?country ?pesticide_amount
            WHERE {{
                ?observation :hasCrop :{crop_entity} ;
                             :hasCountry ?country ;
                             :hasYear  <http://www.semanticweb.org/margaret.gichuhi/ontologies/2024/0/CropYieldOntology#{year_entity}> ;
                             :hasPesticideUsage ?pesticide_usage .
                ?pesticide_usage :hasPesticideAmount ?pesticide_amount .
                FILTER (?pesticide_amount > {threshold_entity})
            }}
        """
    else:
        return "INVALID_QUERY"


def generate_crop_production_query(entities):
    crop_entity = next((entity["text"] for entity in entities if entity["type"] in ["PRODUCT", "ORG", "PERSON"]), None)
    country_entity = next((entity["text"] for entity in entities if entity["type"] == "GPE"), None)
    year_entity = next((entity["text"] for entity in entities if entity["type"] == "DATE"), None)

    if country_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT ?yield
            WHERE {{
                ?observation :hasCrop :{crop_entity} ;
                             :hasCountry :{country_entity} ;
                             :hasYear  <http://www.semanticweb.org/margaret.gichuhi/ontologies/2024/0/CropYieldOntology#{year_entity}> ;
                             :hasYieldInHGHA ?yield .
            }}
        """
    else:
        return "INVALID_QUERY"


def generate_crop_grown_query(entities):
    country_entity = next((entity["text"] for entity in entities if entity["type"] == "GPE"), None)

    if country_entity:
        return f"""
            PREFIX : {_namespace}
            SELECT DISTINCT ?crop
            WHERE {{
                ?observation :hasCrop ?crop ;
                             :hasCountry :{country_entity} .
                             
            }}
        """
    else:
        return "INVALID_QUERY"


# Query data function
def query_data(graph, sparql_query):
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


# Index route
@ui.route("/")
def index():
    return render_template("index.html")


# Search route
@ui.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "GET":
        return render_template("search.html", results=None)

    user_query = request.form.get("query")
    print("User Query:", user_query)

    # Load Turtle file
    g = Graph()
    g.parse("output.ttl",
            format="turtle")

    # Perform Natural Language Understanding (NLU)
    nlu_output = process_nlu(user_query)
    print("NLU Output:", nlu_output)

    # Translate NLU output to SPARQL query using NLQF
    sparql_query = translate_to_sparql(nlu_output)
    print("SPARQL Query:", sparql_query)

    # Query the data
    results = query_data(g, sparql_query)
    print("Results:", results)

    # Prepare results for display
    formatted_results = []
    for result in results:
        formatted_result = {
            "country": result.get("country", {}).get("value", None),
            "crop": result.get("crop", {}).get("value", None),
            "year": result.get("year", {}).get("value", None),
            "yield": result.get("yield", {}).get("value", None),
            "temperature": result.get("temperature", {}).get("value", None),
            "rainfall": result.get("rainfall", {}).get("value", None),
            "pesticide": result.get("pesticide", {}).get("value", None)
        }
        formatted_results.append(formatted_result)


    return render_template("search.html", results=formatted_results)
