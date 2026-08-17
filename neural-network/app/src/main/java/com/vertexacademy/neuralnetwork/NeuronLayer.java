/*
 * @File    : NeuronLayer.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Implementation d'une couche de neurones (NeuronLayer),
 *          adaptee en Java a partir du code Pharo issu du livre
 *          "Practical AI with Pharo" d'Alexandre Bergel :
 *
 *              NeuronLayer>>initializeNbOfNeurons: nbOfNeurons nbOfWeights:
 *                  nbOfWeights using: random
 *                  | weights |
 *                  neurons := (1 to: nbOfNeurons) collect: [ :i |
 *                      weights := (1 to: nbOfWeights) collect: [ :ii | random next * 4 - 2 ].
 *                      Neuron new sigmoid; weights: weights; bias: (random next * 4 - 2) ].
 *                  self learningRate: 0.1
 *
 *          Une couche est composee de plusieurs SigmoidNeuron, chacun
 *          initialise avec des poids et un biais aleatoires tires
 *          dans l'intervalle [-2, 2[ (random * 4 - 2), et un taux
 *          d'apprentissage commun fixe a 0.1.
*/

package com.vertexacademy.neuralnetwork;

import java.util.Random;

public class NeuronLayer {

    /* ==== INSTANCE VARIABLES === */

    private SigmoidNeuron[] neurons;
    private double learningRate;


    /*
    *   Constructeur pratique : cree une couche de nbOfNeurons neurones,
    *   chacun possedant nbOfWeights poids, en utilisant un generateur
    *   aleatoire cree en interne.
    */
    public NeuronLayer(int nbOfNeurons, int nbOfWeights)
    {
        this(nbOfNeurons, nbOfWeights, new Random());
    }

    /*
    *   Constructeur principal, equivalent de la methode Pharo
    *   initializeNbOfNeurons:nbOfWeights:using:
    */
    public NeuronLayer(int nbOfNeurons, int nbOfWeights, Random random)
    {
        this.initializeNbOfNeurons(nbOfNeurons, nbOfWeights, random);
    }

    /*
    *   Initialise le tableau de neurones de la couche : pour chaque
    *   neurone, on genere un tableau de poids aleatoires ainsi qu'un
    *   biais aleatoire, tous deux dans l'intervalle [-2, 2[.
    */
    private void initializeNbOfNeurons(int nbOfNeurons, int nbOfWeights, Random random)
    {
        this.neurons = new SigmoidNeuron[nbOfNeurons];

        for (int i = 0; i < nbOfNeurons; i++)
        {
            double[] weights = new double[nbOfWeights];

            for (int j = 0; j < nbOfWeights; j++)
            {
                weights[j] = random.nextDouble() * 4 - 2;
            }

            SigmoidNeuron neuron = new SigmoidNeuron(nbOfWeights);
            neuron.setWeights(weights);
            neuron.setBias(random.nextDouble() * 4 - 2);

            this.neurons[i] = neuron;
        }

        this.setLearningRate(0.1);
    }

    /*
    *   Fait "avancer" chaque neurone de la couche avec les memes
    *   entrees, et renvoie le tableau des sorties (une par neurone).
    */
    public int[] feed(double[] inputs)
    {
        int[] outputs = new int[this.neurons.length];

        for (int i = 0; i < this.neurons.length; i++)
        {
            outputs[i] = this.neurons[i].feed(inputs);
        }

        return outputs;
    }

    /*[Getter]*/

    public SigmoidNeuron[] getNeurons() {return neurons;}
    public double getLearningRate() {return learningRate;}

    /*[Setter]*/

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;

        for (SigmoidNeuron neuron : this.neurons)
        {
            neuron.setLearningRate(learningRate);
        }
    }


    public static void main(String[] args) {

        NeuronLayer layer = new NeuronLayer(3, 2);

        System.out.println("Couche de " + layer.getNeurons().length
                        + " neurones, learningRate = " + layer.getLearningRate());

        int[] outputs = layer.feed(new double[]{0.0, 1.0});

        System.out.print("Sorties de la couche : [");
        for (int i = 0; i < outputs.length; i++)
        {
            System.out.print(outputs[i] + (i < outputs.length - 1 ? ", " : ""));
        }
        System.out.println("]");
    }
}
