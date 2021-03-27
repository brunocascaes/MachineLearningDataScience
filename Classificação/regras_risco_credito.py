import Orange

base = Orange.data.Table('risco-credito.csv')
base.domain # nome das features

cn2_leaner = Orange.classification.rules.CN2Learner() # geração das regras
classificador = cn2_leaner(base)
# precisou colocar "c#" no arquivo risco-credito

for regras in classificador.rule_list:
    print(regras)
    
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
for i in resultado:
    print(base.domain.class_var.values[i])
# transforma no rótulo correspondente da classe