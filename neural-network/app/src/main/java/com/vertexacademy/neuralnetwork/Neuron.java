/*
 * @File    : Neuron.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Implentation d'une classe Neuron. Elle possede 
 *          des variables d'instance (weights et bias comme mentionne dans le cours),
 *          une sortie output.
 *          Des Methodes publiques :
 *              * Getter :
 *                  * getWeights () : double[]
 *                  * getBias() : double
 *                  * getLearningRate() : double
 *              * Setter :
 *                  * setWeights(double[]),
 *                  * setBias(double)
 *                  * setLearningRate(double)
 *              * feed(double[] inputs) : int 
 *              
*/

package com.vertexacademy.neuralnetwork;

/*
    * Ici nous parlons de Neuron et pas de perceptron car
    * dans la suite du cours cette classe sera etendue et
    * redeclaree comme etant abstraite afin de creer nos
    * differents types de reseaux de neurones.
    *
*/

import java.util.Arrays;
import java.util.Random;


public class Neuron {

    /* ==== INSTANCE VARIABLES === */

    private double[] weights;
    private double bias;
    private double output;
    private double learningRate;


    /*
    *   Constructor pour initialiser un neurone avec un nombre spécifique d'entrées.
    *   Les poids et le biais sont souvent initialisés de manière aléatoire.
    */

    public Neuron(int nbInputs) 
    {
        this.weights = new double[nbInputs];
        Random random = new Random();

        for (int i = 0; i < nbInputs; i++)
        {
            this.weights[i] = random.nextDouble() * 2 - 1;
        }
        this.bias = random.nextDouble() * 2 - 1;

        // Fixer une valeur de coeficient d'apprentissage tres proche
        // de zero
        this.learningRate = 0.1;
    }   

    /*
    *   Fonction d'activation du neuron.
    */
    public int feed(double[] inputs)
    {
        if (inputs.length != weights.length)
            throw new IllegalArgumentException("Le nombre d'entrées ne correspond pas au nombre de poids du neurone.");
        
        double sum = 0.0;

        for (int i = 0; i < inputs.length; i++)
        {
            sum += (weights[i] * inputs[i]);
        }
        
        this.output = sum + bias;
        return (this.output > 0) ? 1 : 0;
    }

    /*
    *   Methode d'entrainement d'un neurone
    */
    public void train(double[] inputs, double desiredOutput) 
    {
        var output = this.feed(inputs);

        // theError represente la difference entre la valeur de sortie 
        // souhaitee et la valeur de sortie actuelle
        var theError = desiredOutput - output;

        // Cette boucle permet d'ajuster et mettre a jour les poids ainsi du perceptron
        for (int i = 0; i < inputs.length; i++)
        {
            var newWeights = this.weights[i] + (this.learningRate * theError * inputs[i]);
            this.weights[i] = newWeights;
        }
        this.bias += (this.learningRate * theError);
    }

    /*[Getter]*/
    
    public double[] getWeights() {return weights;}
    public double getBias() {return bias;}
    public double getLearningRate() {return learningRate;}

    /*[Setter]*/

    public void setWeights(double[] weights) { this.weights = weights;}
    public void setBias(double bias) { this.bias = bias;}
    public void setLearningRate(double learningRate) {this.learningRate = learningRate;}


    public static void main(String[] args) {

        double[] data = {0.0, 1.0};
        Neuron n = new Neuron(2);

        n.setWeights(new double[]{-1.0, -1.0});
        n.setBias(2.0);

        System.out.println("\n\nBefore training...");

        System.out.println("neuron bias = " + n.getBias());
        System.out.println("neuron weights : [" + n.getWeights()[0]
                        + ", " + n.getWeights()[1] + "]");
        
       
        n.train(data, 0);

        System.out.println("\n\nAfter training...");

        System.out.println("neuron bias = " + n.getBias());
        System.out.println("neuron weights : [" + n.getWeights()[0]
                        + ", " + n.getWeights()[1] + "]");
        System.out.println("neuron fires " + n.feed(data));
    }
}
