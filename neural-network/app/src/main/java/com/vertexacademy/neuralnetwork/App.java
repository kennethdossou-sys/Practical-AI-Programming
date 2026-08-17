/*
 * @File    : App.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Classe de demonstration (point d'entree) permettant de :
 *              1) Tester un SigmoidNeuron seul : creation, feed(),
 *                 puis quelques iterations d'entrainement sur une
 *                 porte logique OR pour verifier que la sortie
 *                 continue se rapproche bien des valeurs attendues.
 *              2) Construire une NeuronLayer et verifier que chaque
 *                 neurone de la couche produit bien une sortie.
*/

package com.vertexacademy.neuralnetwork;

public class App {

    public static void main(String[] args) {

        testSigmoidNeuron();
        //testNeuronLayer();
    }

    /*
    *   Teste un SigmoidNeuron seul : on l'initialise avec des poids
    *   et un biais connus, on observe sa sortie avant entrainement,
    *   puis on l'entraine sur la table de verite de la porte OR
    *   afin de verifier que sa sortie continue converge bien vers
    *   les valeurs desirees (proche de 0 ou proche de 1).
    */
    private static void testSigmoidNeuron() {

        System.out.println("=== Test de SigmoidNeuron ===\n");

        SigmoidNeuron neuron = new SigmoidNeuron(2);
        neuron.setWeights(new double[]{0.5, 0.5});
        neuron.setBias(0.0);
        neuron.setLearningRate(0.5);

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[] desiredOutputs = {0, 1, 1, 1}; // porte OR

        System.out.println("Avant entrainement :");
        afficherSorties(neuron, inputs);

        // Entrainement sur plusieurs epoques
        int nbEpochs = 5000;
        for (int epoch = 0; epoch < nbEpochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                neuron.train(inputs[i], desiredOutputs[i]);
            }
        }

        System.out.println("\nAprès " + nbEpochs + " epoques d'entrainement :");
        afficherSorties(neuron, inputs);
        System.out.println();
    }

    /*
    *   Affiche, pour chaque entree, la sortie continue (sigmoide)
    *   ainsi que la sortie binaire (seuillee a 0.5) du neurone.
    */
    private static void afficherSorties(SigmoidNeuron neuron, double[][] inputs) {
        for (double[] input : inputs) {
            int binaryOutput = neuron.feed(input);
            double continuousOutput = neuron.getContinuousOutput();

            System.out.printf("  [%.0f, %.0f] -> sortie continue = %.4f | sortie binaire = %d%n",
                            input[0], input[1], continuousOutput, binaryOutput);
        }
    }

    /*
    *   Construit une NeuronLayer et verifie que chaque neurone de la
    *   couche produit bien une sortie lorsqu'on lui transmet des
    *   entrees. Comme les poids/biais sont initialises aleatoirement,
    *   ce test verifie surtout la bonne construction de la couche et
    *   la propagation correcte a travers tous ses neurones.
    */
    // private static void testNeuronLayer() {

    //     System.out.println("=== Test de NeuronLayer ===\n");

    //     int nbOfNeurons = 4;
    //     int nbOfWeights = 3;

    //     NeuronLayer layer = new NeuronLayer(nbOfNeurons, nbOfWeights);

    //     System.out.println("Couche construite avec " + layer.getNeurons().length
    //                     + " neurones, chacun avec " + nbOfWeights + " poids.");
    //     System.out.println("learningRate de la couche = " + layer.getLearningRate());

    //     double[] sampleInputs = {1.0, 0.0, 1.0};
    //     int[] outputs = layer.feed(sampleInputs);

    //     System.out.print("\nSorties de la couche pour l'entree [1, 0, 1] : [");
    //     for (int i = 0; i < outputs.length; i++) {
    //         System.out.print(outputs[i] + (i < outputs.length - 1 ? ", " : ""));
    //     }
    //     System.out.println("]");

    //     // Detail par neurone : sortie continue + poids/biais utilises
    //     System.out.println("\nDetail par neurone :");
    //     SigmoidNeuron[] neurons = layer.getNeurons();
    //     for (int i = 0; i < neurons.length; i++) {
    //         SigmoidNeuron n = neurons[i];
    //         System.out.printf("  Neurone %d : sortie continue = %.4f | bias = %.4f%n",
    //                         i, n.getContinuousOutput(), n.getBias());
    //     }
    // }
}