###REDE NEURAL DE UM ÚNICO NEURÔNIO PARA FINS DIDÁTICOS###
import math

#Entrada e Saída
input = 0
output_desire = 0

#Peso da Entrada
input_weight = 0.5

#Definição taxa de aprendizagem
learning_rate = 0.01

#Função de ativação
def activation(sum):
    if sum >= 0:
        return 1
    else:
        return 0

print("Entrada", input, "Desejado", output_desire)

#Declaração do Erro como infinito
error = math.inf

#Declaração de um neurônio artificial para correção do input = 0
bias = 1
bias_weight = 0.5

#Declaração variável de iteração para controle de aprendizagem
iteration = 0

#Alteração de peso sobre input até alcançar o erro = 0
while error != 0:
    iteration += 1
    print("########Iteração: ", iteration)
    print("Peso: ", input_weight)
    #Soma da Entrada
    sum = (input * input_weight) + (bias * bias_weight)  # + ... Se houvessem mais neurônios seria somado

    #Ativação
    output = activation(sum)
    print("Saída", output)

    #Cálculo do Erro
    error = output_desire - output
    print("Erro", error)

    #Ajuste do erro
    if error != 0:
        input_weight = input_weight + (learning_rate * input * error)
        bias_weight = bias_weight + (learning_rate * bias * error)
print("Parabéns, a rede aprendeu a resposta!")