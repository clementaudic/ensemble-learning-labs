import os
from pandas import read_csv, cut

# Proving OCCP feature pertinence: Global wage mean: 0.410083 VS engineer subset mean: 0.809428

# Occupations from https://usa.ipums.org/usa/volii/c2ssoccup.shtml
# Management, Business, Science, and Arts 10 -> 500
# Business Operations Specialists 500 -> 800
# Financial Specialists 800 -> 1000
# Computer and Maths 1000 -> 1300
# Architecture and Engineering 1300 -> 1600
# Life, Physical, and Social Science 1600 -> 2000
# Community and Social Services 2000 -> 2100
# Education, Training, and Library 2100 -> 2600
# Arts, Design, Entertainment, Sports, and Media 2600 -> 3000
# Literature and Languages -> 2600
# Healthcare Practitioners and Technical 3000 -> 3600
# Healthcare Support 3600 -> 3700
# Protective Service 3700 -> 4000
# Food Preparation and Serving 4000 -> 4200
# Building and Grounds Cleaning and Maintenance 4200 -> 4300
# Personal Care and Service 4300 -> 4700
# Sales and Related 4700 -> 5000
# Office and Administrative Support 5000 -> 6000
# Farming, Fishing, and Forestry 6000 -> 6200
# Construction and Extraction 6200 -> 6800
# Extraction Workers 6800 -> 7000
# Installation, Maintenance, and Repair Workers 7000 -> 7700
# Production 7700 -> 9000
# Transportation and Material Moving 9000 -> 9800
# Military Specific 9800 -> 10000

occupation_bins = [
    0, 500, 800, 1000, 1300, 1600, 2000, 2100, 2600, 3000, 3600, 3700, 4000, 4200, 4300, 4700, 5000,
    6000, 6200, 6800,7000, 7700, 9000, 9800, 10000
]

occupation_labels = [
    "Management, Business, Science, and Arts",
    "Business Operations Specialists",
    "Financial Specialists",
    "Computer and Maths",
    "Architecture and Engineering",
    "Life, Physical, and Social Science",
    "Community and Social Services",
    "Education, Training, and Library",
    "Arts, Design, Entertainment, Sports, and Media",
    "Healthcare Practitioners and Technical",
    "Healthcare Support",
    "Protective Service",
    "Food Preparation and Serving",
    "Building and Grounds Cleaning and Maintenance",
    "Personal Care and Service",
    "Sales and Related",
    "Office and Administrative Support",
    "Farming, Fishing, and Forestry",
    "Construction and Extraction",
    "Extraction Workers",
    "Installation, Maintenance, and Repair Workers",
    "Production",
    "Transportation and Material Moving",
    "Military Specific"
]

def map_pobp(code):

    # Places of birth

    # Northern Europe 106-108, 118-119, 121, 127, 135-136, 138-145 
    # Southern Europe 115-116, 120, 124, 129-131, 133-134, 146
    # Earthern Europe 100, 104-105, 117, 128, 132, 147-157, 160, 162 -165, 167-168
    # Eastern Asia 207, 209, 215, 217, 220-221, 225, 228, 232, 240
    # South Central Asia 200, 202-203, 210, 212, 218-219, 227, 229, 231, 238, 241, 244, 246
    # South Eastern Asia 204-206, 211, 223, 226, 233, 236-237, 242, 247, 250
    # Western Asia 158-159, 161, 201, 208, 213-214, 216, 222, 224, 230, 234-235, 239, 243, 245, 248
    # Northern America 001-059, 061-099, 300-302, 304-309
    # Latin America 303, 310-399
    # Eastern Africa 404, 406, 411, 413, 416-418, 422, 426-427, 431-432, 435, 437, 441-442, 445-446, 448, 453, 455, 457, 460-461, 463
    # Middle Africa 401, 407, 409-410, 412, 415, 419, 443, 459
    # Northern Africa 400, 414, 430, 436, 451, 456, 458
    # Southern Africa 403, 428, 438, 449, 452
    # Western Africa 402, 405, 408, 420-421, 423-425, 429, 433-434, 439-440, 444, 447, 450, 454
    # Oceania (Australia and New-Zealand Subregion) 501-502, 506-507, 515, 517 
    # Oceania (others) 060, 503-505, 508-514, 516, 518-553

    regions = {
        "western_europe": list(range(101, 104)) + [109, 110, 122, 123, 125, 126, 137], # western_europe
        "northern_europe": list(range(106, 109)) + [118, 119, 121, 127] + list(range(135, 137)) + list(range(138, 146)),
        "southern_europe": [115, 116, 120, 124] + list(range(129, 132)) + [133, 134, 146],
        "earthern_europe": [100, 104, 105, 117, 128, 132] + list(range(147, 158)) + [160] + list(range(162, 166)) + [167, 168],
        "eastern_asia": [207, 209, 215, 217] + list(range(220, 222)) + [225, 228, 232, 240],
        "south_central_asia": [200, 202, 203, 210, 212, 218, 219, 227, 229, 231, 238, 241, 244, 246],
        "south_eastern_asia": list(range(204, 207)) + [211, 223, 226, 233, 236, 237, 242, 247, 250],
        "western_asia": list(range(158, 160)) + [161, 201, 208, 213, 214, 216, 222, 224, 230, 234, 235, 239, 243, 245, 248],
        "northern_america": list(range(1, 60)) + list(range(61, 100)) + list(range(300, 303)) + list(range(304, 310)),
        "latin_america": [303] + list(range(310, 400)),
        "eastern_africa": [404, 406, 411, 413, 416, 417, 418, 422, 426, 427, 431, 432, 435, 437, 441, 442, 445, 446, 448, 453, 455, 457, 460, 461, 463],
        "middle_africa": [401, 407, 409, 410, 412, 415, 419, 443, 459],
        "northern_africa": [400, 414, 430, 436, 451, 456, 458],
        "southern_africa": [403, 428, 438, 449, 452],
        "western_africa": [402, 405, 408, 420, 421, 423, 424, 425, 429, 433, 434, 439, 440, 444, 447, 450, 454],
        "oceania": [60] + list(range(500, 554))
    }
    
    for region_name, region_codes in regions.items():
        if code in region_codes:
            return region_name

def group_features(dataset_names,features_to_group_names):
    for dataset_name in dataset_names:
        if "_ca_" in dataset_name:
            state_name = "california"
        if "_co_" in dataset_name:
            state_name = "colorado"
        if "_ne_" in dataset_name:
            state_name = "nevada"
        features_df = read_csv(os.path.join("data",state_name,dataset_name))
        if "OCCP" in features_to_group_names:
            features_df["OCCP"] = cut(features_df["OCCP"],bins = occupation_bins, labels = occupation_labels)
        if "POBP" in features_to_group_names:
            features_df["POBP"] = features_df["POBP"].apply(map_pobp)
        features_df.to_csv(os.path.join("data",state_name,dataset_name+"_with_"+"_and_".join(features_to_group_names)+"_regrouped"),index=False)

# "alt_acsincome_ca_features_85.csv"
# "acsincome_co_allfeatures.csv"
# "acsincome_ne_allfeatures.csv"
group_features(["acsincome_co_allfeatures.csv","acsincome_ne_allfeatures.csv"],["POBP"])