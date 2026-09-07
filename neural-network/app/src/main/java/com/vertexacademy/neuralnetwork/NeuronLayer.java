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
    private NeuronLayer previousLayer;
    private NeuronLayer nextLayer;


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
    *   Constructeur principal
    */
    public NeuronLayer(int nbOfNeurons, int nbOfWeights, Random random)
    {
        this.initializeNbOfNeurons(nbOfNeurons, nbOfWeights, random);
    }

    /**
     * 
     * @return true si la couche est la couche de sortie. La couche de sortie
     * est definie ssi nextLayer est nulle.
    */

    public boolean isOutputLayer() {
        return nextLayer == null;
    }

    /*[Getter]*/

    /**
     * 
     * @return double[] Un tableau de Sigmoid neurons qui compose la couche
     */
    public SigmoidNeuron[] getNeurons() {return neurons;}

    /**
     * 
     * @return double le taux d'apprentissage de la couche
    */
    public double getLearningRate() {return learningRate;}

    public NeuronLayer getPreviousLayer() {return previousLayer;}
    public NeuronLayer getNextLayer() {return nextLayer;}


    /*[Setter]*/

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;

        for (SigmoidNeuron neuron : this.neurons)
        {
            neuron.setLearningRate(learningRate);
        }
    }

    /**
     * @brief Permet de chainer les couches entre elles.
     * @param layer
    */

    public void setPreviousLayer(NeuronLayer layer) {
        previousLayer = layer;
    }

    /**
     * @brief Permet de chainer les couches entre elles.
     * @param layer
     */
    public void setNextLayer(NeuronLayer layer) {
        nextLayer = layer;
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

        if(this.isOutputLayer()) {
            return outputs;
        }
        else{
            double[] newOutputs = new double[outputs.length];

            // Cast int to double before send to the next layer
            for (int i = 0; i < outputs.length; i++) {
                newOutputs[i] = ((double)outputs[i]);
            }

            return this.nextLayer.feed(newOutputs);
        }
    }

    
    public void train(double[] inputs, double[] desiredOutputs) {

        if (desiredOutputs.length != this.neurons.length) {
            throw new IllegalArgumentException(
                    "Le nombre de sorties desirees doit correspondre " + 
                    "au nombre de neurones de la couche.");
        }

        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i].train(inputs, desiredOutputs[i]);
        }
    }


    public static void main(String[] args) {

        NeuronLayer layer = new NeuronLayer(4, 2);

        System.out.println("Couche de " + layer.getNeurons().length
                        + " neurones, learningRate = " + layer.getLearningRate());

        int[] outputs = layer.feed(new double[]{0.0, 1.0});
        
        // Affichage des poids de chaque neurone
        System.out.println("Affichage des poids de chaque neuron de la couche ");
        System.out.println();

        for (int i = 0; i < layer.neurons.length; i++)
        {
            System.out.printf("neuron[%d]\nw[0] = %.4f \t w[1] = %.4f\t continiousOutput= %.4f\tbias = %.4f", 
                        i, layer.neurons[i].getWeights()[0], 
                        layer.neurons[i].getWeights()[1],
                        layer.neurons[i].getContinuousOutput(),
                        layer.neurons[i].getBias());
            System.out.println();
        }

        System.out.println();
        System.out.print("Sorties de la couche : [");


        for (int i = 0; i < outputs.length; i++)
        {
            System.out.print(outputs[i] + (i < outputs.length - 1 ? ", " : ""));
        }
        System.out.println("]");

    }
}
