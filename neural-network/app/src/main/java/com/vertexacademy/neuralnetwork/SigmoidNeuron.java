/*
 * @File    : SigmoidNeuron.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Sous-classe de Neuron qui remplace la fonction d'activation
 *          en escalier (step function) par une fonction sigmoide.
 *          Contrairement au Neuron de base (perceptron), la sortie interne
 *          (output) est ici une valeur continue comprise entre 0 et 1,
 *          et non plus une simple valeur binaire.
 *
 *          feed(double[]) renvoie toujours un entier (0 ou 1), obtenu en
 *          seuillant la sortie continue a 0.5, afin de rester compatible
 *          avec l'interface de la classe Neuron.
 *
 *          train(double[], double) est egalement redefinie afin d'utiliser
 *          la derivee de la sigmoide dans la mise a jour des poids
 *          (descente de gradient), ce qui est plus adapte a une fonction
 *          d'activation continue que la regle d'apprentissage du perceptron.
*/

package com.vertexacademy.neuralnetwork;

public class SigmoidNeuron extends Neuron {

    public SigmoidNeuron(int nbInputs)
    {
        super(nbInputs);
    }

    /*
    *   Fonction sigmoide : ecrase n'importe quelle valeur reelle
    *   dans l'intervalle ]0, 1[.
    */
    private double sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /*
    *   Derivee de la sigmoide, exprimee directement a partir de la
    *   sortie deja calculee (sigmoid(x) * (1 - sigmoid(x))).
    */
    private double sigmoidDerivative(double sigmoidOutput)
    {
        return sigmoidOutput * (1 - sigmoidOutput);
    }

    /*
    *   Fonction d'activation du neurone, remplace la function en escalier
    *   du Neuron de base (Perceptron) par une sigmoide.
    */
    @Override
    public int feed(double[] inputs)
    {
        if (inputs.length != weights.length)
            throw new IllegalArgumentException("Le nombre d'entrées ne correspond pas au nombre de poids du neurone.");

        double sum = 0.0;

        for (int i = 0; i < inputs.length; i++)
        {
            sum += (weights[i] * inputs[i]);
        }

        this.output = sigmoid(sum + bias);
        return (this.output > 0.5) ? 1 : 0;
    }

    /*
    *   Methode d'entrainement adaptee a la sigmoide : la mise a jour
    *   des poids utilise le gradient de l'erreur (erreur * derivee de
    *   la sigmoide), plutot que directement l'erreur comme dans le
    *   Neuron de base.
    */
    @Override
    public void train(double[] inputs, double desiredOutput)
    {
        this.feed(inputs);

        // theError represente la difference entre la valeur de sortie
        // souhaitee et la sortie continue actuelle du neurone
        var theError = desiredOutput - this.output;
        var gradient = theError * sigmoidDerivative(this.output);

        for (int i = 0; i < inputs.length; i++)
        {
            var newWeights = this.weights[i] + (this.learningRate * gradient * inputs[i]);
            this.weights[i] = newWeights;
        }
        this.bias += (this.learningRate * gradient);
    }

    /*
    *   Permet de recuperer la sortie continue (avant seuillage),
    *   utile notamment pour la couche NeuronLayer et pour la
    *   retropropagation.
    */
    public double getContinuousOutput()
    {
        return this.output;
    }
}
