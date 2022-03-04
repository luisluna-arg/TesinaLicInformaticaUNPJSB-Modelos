import json

def importFile(samples, file_path, direction):
    f = open(file_path)
    data = json.load(f)
    f.close()

    for i in data: 
        sample = list()
        sample.append(i['eegPower']['delta'])
        sample.append(i['eegPower']['theta'])
        sample.append(i['eegPower']['lowAlpha'])
        sample.append(i['eegPower']['highAlpha'])
        sample.append(i['eegPower']['lowBeta'])
        sample.append(i['eegPower']['highBeta'])
        sample.append(i['eegPower']['lowGamma'])
        sample.append(i['eegPower']['highGamma'])
        sample.append(direction)
        samples.append(sample)
        
def load_files():
    columns = list()

    columns.append('delta')
    columns.append('theta')
    columns.append('lowAlpha')
    columns.append('highAlpha')
    columns.append('lowBeta')
    columns.append('highBeta')
    columns.append('lowGamma')
    columns.append('highGamma')
    columns.append('direction')

    ###############################################################
    ###############################################################
    ###############################################################

    data = list()

    abajoIx = 1
    arribaIx = 2
    izquierdaIx = 3
    derechaIx = 4

    # base_path = 'data/2022.2.22/'
    base_path = 'data/Full/'
    # base_path = 'data/generated/'

    importFile(data, base_path + 'ABAJO.json', abajoIx)
    importFile(data, base_path + 'ARRIBA.json', arribaIx)
    importFile(data, base_path + 'IZQUIERDA.json', izquierdaIx)
    importFile(data, base_path + 'DERECHA.json', derechaIx)

    return columns, data