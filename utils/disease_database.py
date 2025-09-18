# Plant Disease Information Database
DISEASE_DATABASE = {
    # Apple Diseases
    "Apple___Apple_scab": {
        "common_name": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "severity": "Moderate to High",
        "description": "Apple scab is a fungal disease causing dark, scabby lesions on leaves and fruit.",
        "symptoms": [
            "Dark, olive-green to black lesions on leaves",
            "Scabby lesions on fruit",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "causes": [
            "Cool, wet weather conditions",
            "High humidity (above 95%)",
            "Poor air circulation",
            "Overhead irrigation"
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil)",
            "Remove fallen leaves and debris",
            "Prune for better air circulation",
            "Use resistant apple varieties"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Ensure good drainage",
            "Avoid overhead watering",
            "Regular sanitation practices"
        ],
        "economic_impact": "Can reduce yield by 50-70% in severe cases",
        "season": "Spring and early summer",
        "spread": "Wind-dispersed spores during rain",
        "management_cost": "Medium"
    },
    
    "Apple___Black_rot": {
        "common_name": "Apple Black Rot",
        "scientific_name": "Botryosphaeria obtusa",
        "severity": "High",
        "description": "Black rot is a serious fungal disease affecting apple fruit and wood.",
        "symptoms": [
            "Brown to black circular lesions on fruit",
            "Mummified fruit ('mummies')",
            "Cankers on branches and trunk",
            "Leaf spots with yellow halos"
        ],
        "causes": [
            "Warm, humid conditions",
            "Wounded or stressed trees",
            "Poor orchard sanitation",
            "Dense canopy with poor air circulation"
        ],
        "treatment": [
            "Remove mummified fruit and infected wood",
            "Apply fungicides (propiconazole, thiophanate-methyl)",
            "Improve orchard sanitation",
            "Prune infected branches"
        ],
        "prevention": [
            "Regular pruning for air circulation",
            "Remove all mummies and debris",
            "Avoid tree stress through proper irrigation",
            "Use preventive fungicide sprays"
        ],
        "economic_impact": "Can cause complete fruit loss in untreated orchards",
        "season": "Mid to late growing season",
        "spread": "Rain splash and wind",
        "management_cost": "High"
    },
    
    "Apple___Cedar_apple_rust": {
        "common_name": "Cedar Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "severity": "Moderate",
        "description": "A fungal disease requiring both cedar and apple trees to complete its lifecycle.",
        "symptoms": [
            "Bright orange spots on apple leaves",
            "Yellow-orange lesions on fruit",
            "Tube-like structures on leaf undersides",
            "Early leaf drop"
        ],
        "causes": [
            "Presence of cedar/juniper trees nearby",
            "Wet spring weather",
            "Spores released from cedar galls",
            "Wind dispersal of spores"
        ],
        "treatment": [
            "Apply protective fungicides in spring",
            "Remove nearby cedar trees if possible",
            "Use resistant apple varieties",
            "Improve air circulation"
        ],
        "prevention": [
            "Plant resistant apple varieties",
            "Remove cedar trees within 1-2 miles",
            "Apply preventive fungicides",
            "Monitor weather conditions"
        ],
        "economic_impact": "Moderate yield loss and quality reduction",
        "season": "Spring to early summer",
        "spread": "Wind-dispersed spores from cedar trees",
        "management_cost": "Medium"
    },
    
    "Apple___healthy": {
        "common_name": "Healthy Apple",
        "scientific_name": "Malus domestica",
        "severity": "None",
        "description": "Healthy apple leaves showing normal green coloration and structure.",
        "symptoms": [
            "Uniform green coloration",
            "No spots or lesions",
            "Normal leaf size and shape",
            "Good leaf attachment"
        ],
        "causes": ["Optimal growing conditions"],
        "treatment": ["Continue current management practices"],
        "prevention": [
            "Maintain proper nutrition",
            "Ensure adequate water supply",
            "Regular monitoring for diseases",
            "Good orchard sanitation"
        ],
        "economic_impact": "Optimal fruit production expected",
        "season": "All growing seasons",
        "spread": "Not applicable",
        "management_cost": "Low maintenance"
    },
    
    # Tomato Diseases
    "Tomato___Bacterial_spot": {
        "common_name": "Tomato Bacterial Spot",
        "scientific_name": "Xanthomonas spp.",
        "severity": "High",
        "description": "Bacterial disease causing leaf spots and fruit lesions in tomatoes.",
        "symptoms": [
            "Small, dark brown leaf spots",
            "Yellow halos around spots",
            "Raised, scabby fruit lesions",
            "Defoliation in severe cases"
        ],
        "causes": [
            "Warm, humid weather",
            "Overhead irrigation",
            "Contaminated seeds or transplants",
            "Wounds from insects or tools"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Improve air circulation",
            "Use drip irrigation"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Avoid overhead watering",
            "Rotate crops",
            "Disinfect tools between plants"
        ],
        "economic_impact": "Can reduce yield by 30-50%",
        "season": "Warm, humid periods",
        "spread": "Water splash, contaminated tools",
        "management_cost": "Medium to High"
    },
    
    "Tomato___Early_blight": {
        "common_name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate to High",
        "description": "Common fungal disease causing characteristic target spot lesions.",
        "symptoms": [
            "Circular brown spots with concentric rings",
            "Target-like appearance of lesions",
            "Lower leaves affected first",
            "Collar rot at soil line"
        ],
        "causes": [
            "Warm, humid conditions",
            "Plant stress or poor nutrition",
            "Dense foliage with poor air circulation",
            "Splashing water"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, azoxystrobin)",
            "Remove infected lower leaves",
            "Improve plant nutrition",
            "Ensure proper spacing"
        ],
        "prevention": [
            "Use resistant varieties",
            "Maintain proper plant nutrition",
            "Ensure good air circulation",
            "Water at soil level"
        ],
        "economic_impact": "Can reduce yield by 20-40%",
        "season": "Mid to late growing season",
        "spread": "Wind and rain splash",
        "management_cost": "Medium"
    },
    
    "Tomato___Late_blight": {
        "common_name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "Very High",
        "description": "Devastating disease that can destroy entire crops rapidly.",
        "symptoms": [
            "Large, irregular brown lesions",
            "Water-soaked appearance",
            "White fuzzy growth on leaf undersides",
            "Rapid plant death"
        ],
        "causes": [
            "Cool, wet weather",
            "High humidity",
            "Poor air circulation",
            "Infected potato debris"
        ],
        "treatment": [
            "Apply protectant fungicides immediately",
            "Remove infected plants entirely",
            "Improve drainage",
            "Emergency spray programs"
        ],
        "prevention": [
            "Use resistant varieties",
            "Ensure excellent drainage",
            "Apply preventive fungicides",
            "Monitor weather conditions"
        ],
        "economic_impact": "Can cause 100% crop loss",
        "season": "Cool, wet periods",
        "spread": "Airborne spores, very rapid",
        "management_cost": "Very High"
    },
    
    "Tomato___healthy": {
        "common_name": "Healthy Tomato",
        "scientific_name": "Solanum lycopersicum",
        "severity": "None",
        "description": "Healthy tomato leaves with normal appearance and function.",
        "symptoms": [
            "Dark green, uniform coloration",
            "No spots, lesions, or discoloration",
            "Normal leaf shape and size",
            "Strong attachment to stem"
        ],
        "causes": ["Optimal growing conditions and care"],
        "treatment": ["Continue current practices"],
        "prevention": [
            "Maintain consistent watering",
            "Provide adequate nutrition",
            "Monitor for early disease signs",
            "Practice good garden hygiene"
        ],
        "economic_impact": "Expected normal yield",
        "season": "Throughout growing season",
        "spread": "Not applicable",
        "management_cost": "Regular maintenance"
    },
    
    # Add more diseases as needed...
    "default_disease": {
        "common_name": "Unknown Disease",
        "scientific_name": "Unidentified pathogen",
        "severity": "Unknown",
        "description": "Disease not found in database. Consult local agricultural extension service.",
        "symptoms": ["Consult expert for diagnosis"],
        "causes": ["Unknown - requires expert diagnosis"],
        "treatment": ["Seek professional agricultural advice"],
        "prevention": ["Maintain good plant health practices"],
        "economic_impact": "Unknown - varies by condition",
        "season": "Varies",
        "spread": "Unknown",
        "management_cost": "Varies"
    }
}

# Treatment recommendations by severity
SEVERITY_RECOMMENDATIONS = {
    "None": {
        "urgency": "No action needed",
        "action": "Continue monitoring",
        "color": "green"
    },
    "Low": {
        "urgency": "Monitor closely",
        "action": "Preventive measures recommended",
        "color": "yellow"
    },
    "Moderate": {
        "urgency": "Take action within 1-2 weeks",
        "action": "Begin treatment program",
        "color": "orange"
    },
    "High": {
        "urgency": "Take immediate action",
        "action": "Begin intensive treatment",
        "color": "red"
    },
    "Very High": {
        "urgency": "EMERGENCY - Act within 24-48 hours",
        "action": "Emergency treatment required",
        "color": "darkred"
    }
}

# General plant care recommendations
GENERAL_RECOMMENDATIONS = {
    "healthy": [
        "Continue current care practices",
        "Monitor regularly for early disease signs",
        "Maintain proper nutrition and watering",
        "Ensure good air circulation",
        "Practice garden sanitation"
    ],
    "diseased": [
        "Isolate affected plants if possible",
        "Improve air circulation around plants",
        "Adjust watering practices (avoid overhead watering)",
        "Consider fungicide/bactericide application",
        "Remove and dispose of infected material properly"
    ]
}

def get_disease_info(class_name: str) -> dict:
    """
    Get comprehensive disease information from the database.
    
    Args:
        class_name (str): The predicted class name
        
    Returns:
        dict: Comprehensive disease information
    """
    return DISEASE_DATABASE.get(class_name, DISEASE_DATABASE["default_disease"])

def get_severity_info(severity: str) -> dict:
    """Get severity-based recommendations."""
    return SEVERITY_RECOMMENDATIONS.get(severity, SEVERITY_RECOMMENDATIONS["Moderate"])

def format_disease_report(disease_info: dict, confidence: float) -> str:
    """
    Format a comprehensive disease report.
    
    Args:
        disease_info (dict): Disease information from database
        confidence (float): Prediction confidence
        
    Returns:
        str: Formatted disease report
    """
    severity = disease_info.get('severity', 'Unknown')
    severity_info = get_severity_info(severity)
    
    report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         🌱 PLANT DISEASE ANALYSIS REPORT 🌱                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 DIAGNOSIS SUMMARY
├─ Disease: {disease_info['common_name']}
├─ Scientific Name: {disease_info['scientific_name']}
├─ Confidence: {confidence:.1%}
├─ Severity Level: {severity}
└─ Urgency: {severity_info['urgency']}

📝 DESCRIPTION
{disease_info['description']}

🔍 SYMPTOMS TO LOOK FOR
"""
    for i, symptom in enumerate(disease_info['symptoms'], 1):
        report += f"   {i}. {symptom}\n"

    report += f"""
🦠 CAUSES & CONDITIONS
"""
    for i, cause in enumerate(disease_info['causes'], 1):
        report += f"   {i}. {cause}\n"

    report += f"""
💊 TREATMENT RECOMMENDATIONS
"""
    for i, treatment in enumerate(disease_info['treatment'], 1):
        report += f"   {i}. {treatment}\n"

    report += f"""
🛡️ PREVENTION STRATEGIES
"""
    for i, prevention in enumerate(disease_info['prevention'], 1):
        report += f"   {i}. {prevention}\n"

    report += f"""
📈 ECONOMIC & MANAGEMENT INFO
├─ Economic Impact: {disease_info['economic_impact']}
├─ Peak Season: {disease_info['season']}
├─ Spread Method: {disease_info['spread']}
└─ Management Cost: {disease_info['management_cost']}

⚠️ IMMEDIATE ACTION REQUIRED
{severity_info['action']}

═══════════════════════════════════════════════════════════════════════════════
🔬 This analysis is AI-generated. For severe cases or uncertain diagnoses,
   consult with local agricultural extension services or plant pathologists.
═══════════════════════════════════════════════════════════════════════════════
"""
    return report