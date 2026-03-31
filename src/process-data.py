import xmltodict
import os

def get_row(abstract: list[dict]):
	row = ""

	for x in abstract:
		if x['@Label'] == 'IMPRESSION':
			row += f"{x.get("#text", "")}$"
		elif x['@Label'] == 'FINDINGS':
			row += f"{x.get("#text", "")}\n"
	return row

if __name__ == '__main__':
	files = os.listdir('data/ecgen-radiology')

	with open('data/processed_data.csv', 'w') as fp:
		fp.write('IMPRESSION$FINDINGS\n')

	for data_file in files:
		with open(f'data/ecgen-radiology/{data_file}', 'rb') as fp:
			xml_dict = xmltodict.parse(fp)
		abstract = xml_dict['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']
		row = get_row(abstract)
		with open('data/processed_data.csv', 'a') as fp:
			fp.write(row)
