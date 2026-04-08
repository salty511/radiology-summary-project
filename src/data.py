import xmltodict
import os
import re

def get_row(abstract: list[dict]):
    impression = ""
    findings = ""
    
    for x in abstract:
        if x['@Label'] == 'IMPRESSION':
            impression = x.get("#text", "")
        elif x['@Label'] == 'FINDINGS':
            findings = x.get("#text", "")

    # If either record is not present, ignore row
    if(impression == "" or findings == ""):
        return None

    return f"{impression}${findings}\n"

def get_row_clean(abstract: list[dict], redactedInfo=True, trailingFullStop=True, numberedLists=True):
    impression = ""
    findings = ""
    
    for x in abstract:
        if x['@Label'] == 'IMPRESSION':
            impression = x.get("#text", "")
        elif x['@Label'] == 'FINDINGS':
            findings = x.get("#text", "")

    # If either record is not present, ignore row
    if(impression == "" or findings == ""):
        return None

    if(redactedInfo):
        # Remove redacted info, match string of 2 or more Xs i.e. XXXX
        if re.search(r"X{2,}", impression) or re.search(r"X{2,}", findings):
            return None
    
    if(trailingFullStop):
        # Remove trailing fullstop and space pattern i.e. No acute cardiopulmonary abnormality. .
        impression = re.sub(r"(\.\s+)+\.", ".", impression)
        findings = re.sub(r"(\.\s+)+\.", ".", findings)

    if(numberedLists):
        # Standardise numbered lists to just strings of sentences i.e. 1. No acute pulmonary abnormality. -> No acute pulmonary abnormality.
        impression = re.sub(r"(^|\.\s)\d+\.\s*", r"\1", impression)
        findings = re.sub(r"(^|\.\s)\d+\.\s*", r"\1", findings)

    return f"{impression}${findings}\n"

def run_data_pipeline(file: str):
    files = os.listdir('data/ecgen-radiology')
    print(files[:10])

    file = "processed_data_clean"

    with open(f"data/{file}.csv", 'w') as fp:
        fp.write('IMPRESSION$FINDINGS\n')

    for data_file in files:
        with open(f'data/ecgen-radiology/{data_file}', 'rb') as fp:
            xml_dict = xmltodict.parse(fp)
        abstract = xml_dict['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']

        # Clean or Raw data files for testing
        if(file == "processed_data"):
            row = get_row(abstract)
        else:
            row = get_row_clean(abstract, redactedInfo=True, trailingFullStop=True, numberedLists=True)
        if(row is not None):
            with open(f"data/{file}.csv", 'a') as fp:
                fp.write(row)
