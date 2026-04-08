import xmltodict
import os
import re

def get_row(abstract: list[dict]):
	row = ""

	for x in abstract:
		if x['@Label'] == 'IMPRESSION':
			row += f"{x.get("#text", "")}$"
		elif x['@Label'] == 'FINDINGS':
			row += f"{x.get("#text", "")}\n"
	return row

def get_row_remove_redacted_info(abstract: list[dict]):
    impression = ""
    findings = ""
    for x in abstract:
        if x['@Label'] == 'IMPRESSION':
            impression = x.get("#text", "")
        elif x['@Label'] == 'FINDINGS':
            findings = x.get("#text", "")

    if re.search(r"X{2,}", impression) or re.search(r"X{2,}", findings):
        return None

    return f"{impression}${findings}\n"


if __name__ == '__main__':
	files = os.listdir('data/ecgen-radiology')

	file = "processed_data_no_redacted_info"

	with open(f"data/{file}.csv", 'w') as fp:
		fp.write('IMPRESSION$FINDINGS\n')

	for data_file in files:
		with open(f'data/ecgen-radiology/{data_file}', 'rb') as fp:
			xml_dict = xmltodict.parse(fp)
		abstract = xml_dict['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']
		if(file == "processed_data"):
			row = get_row(abstract)
		else:
			row = get_row_remove_redacted_info(abstract)
		if(row is not None):
			with open(f"data/{file}.csv", 'a') as fp:
				fp.write(row)
