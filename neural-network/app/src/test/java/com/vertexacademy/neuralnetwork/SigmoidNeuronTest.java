/*
 * @File    : SigmoidNeuronTest.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Tests unitaires de la classe SigmoidNeuron.
 *          On y verifie :
 *              * que la sortie continue est bien une sigmoide
 *                (comprise entre 0 et 1, valeur exacte pour des cas connus),
 *              * que le seuillage a 0.5 produit la bonne sortie binaire,
 *              * que getContinuousOutput() renvoie bien la derniere
 *                sortie calculee par feed(),
 *              * que l'exception est toujours levee en cas de mauvais
 *                nombre d'entrees (comportement herite de Neuron),
 *              * que l'entrainement (train) fait converger le neurone
 *                sur une table de verite logique (OR).
*/
package com.vertexacademy.neuralnetwork;

import org.junit.Test;
import static org.junit.Assert.*;

public class SigmoidNeuronTest {

    // Marge d'erreur acceptee pour les comparaisons de double
    private static final double DELTA = 1e-4;

    /*
    *   La methode testSigmoidZeroSum prouve que lorsque la somme
    *   ponderee (poids * entrees + biais) vaut 0, la sortie continue
    *   du neurone vaut exactement 0.5 (sigmoid(0) = 0.5), et que la
    *   sortie binaire associee est donc 0 (0.5 n'est pas strictement
    *   superieur a 0.5).
    */
    @Test
    public void testSigmoidZeroSum() {

        SigmoidNeuron n = new SigmoidNeuron(2);
        n.setWeights(new double[]{0, 0});
        n.setBias(0);

        int binaryOutput = n.feed(new double[]{1, 1});

        assertEquals("sigmoid(0) doit valoir 0.5", 0.5, n.getContinuousOutput(), DELTA);
        assertEquals("sortie binaire doit etre 0 quand la sortie continue vaut 0.5", 0, binaryOutput);
    }

    /*
    *   La methode testSigmoidFireOne prouve que pour une somme
    *   ponderee fortement positive, la sortie continue se rapproche
    *   de 1 et que le neurone "s'active" (sortie binaire = 1).
    */
    @Test
    public void testSigmoidFireOne() {

        SigmoidNeuron n = new SigmoidNeuron(1);
        n.setWeights(new double[]{10});
        n.setBias(0);

        int binaryOutput = n.feed(new double[]{1});

        assertTrue("sortie continue doit etre proche de 1", n.getContinuousOutput() > 0.99);
        assertEquals("neuron should fire one", 1, binaryOutput);
    }

    /*
    *   La methode testSigmoidFireZero prouve que pour une somme
    *   ponderee fortement negative, la sortie continue se rapproche
    *   de 0 et que le neurone ne "s'active" pas (sortie binaire = 0).
    */
    @Test
    public void testSigmoidFireZero() {

        SigmoidNeuron n = new SigmoidNeuron(1);
        n.setWeights(new double[]{-10});
        n.setBias(0);

        int binaryOutput = n.feed(new double[]{1});

        assertTrue("sortie continue doit etre proche de 0", n.getContinuousOutput() < 0.01);
        assertEquals("neuron should fire zero", 0, binaryOutput);
    }

    /*
    *   La methode testContinuousOutputAlwaysBetweenZeroAndOne prouve
    *   que quelles que soient les entrees, poids et biais, la sortie
    *   continue reste toujours dans l'intervalle [0, 1].
    */
    @Test
    public void testContinuousOutputAlwaysBetweenZeroAndOne() {

        SigmoidNeuron n = new SigmoidNeuron(3);
        n.setWeights(new double[]{5, -3.5, 2});
        n.setBias(-1);

        n.feed(new double[]{1, 1, 0});

        double output = n.getContinuousOutput();
        assertTrue("la sortie continue doit etre >= 0", output >= 0.0);
        assertTrue("la sortie continue doit etre <= 1", output <= 1.0);
    }

    /*
    *   La methode testGetContinuousOutputMatchesFeed prouve que
    *   getContinuousOutput() renvoie bien la meme valeur que celle
    *   utilisee en interne par feed() pour calculer la sortie binaire.
    */
    @Test
    public void testGetContinuousOutputMatchesFeed() {

        SigmoidNeuron n = new SigmoidNeuron(2);
        n.setWeights(new double[]{0.5, -0.5});
        n.setBias(0.2);

        n.feed(new double[]{2, 1});

        double expectedSum = (0.5 * 2) + (-0.5 * 1) + 0.2;
        double expectedOutput = 1.0 / (1.0 + Math.exp(-expectedSum));

        assertEquals("getContinuousOutput doit correspondre au calcul manuel de la sigmoide",
                        expectedOutput, n.getContinuousOutput(), DELTA);
    }

    /*
    *   La methode testSigmoidWrongFeeding prouve que, comme pour la
    *   classe Neuron, une exception est levee si le nombre d'entrees
    *   ne correspond pas au nombre de poids du neurone.
    */
    @Test
    public void testSigmoidWrongFeeding() {

        SigmoidNeuron n = new SigmoidNeuron(2);
        n.setWeights(new double[]{1, 1});
        n.setBias(0);

        assertThrows(IllegalArgumentException.class, () -> {
            n.feed(new double[]{1, 1, 1});
        });
    }

    /*
    *   La methode testTrainReducesError prouve qu'apres un
    *   entrainement, la sortie continue du neurone se rapproche de la
    *   sortie desiree (l'erreur diminue par rapport a l'erreur
    *   initiale avant entrainement).
    */
    @Test
    public void testTrainReducesError() {

        SigmoidNeuron n = new SigmoidNeuron(2);
        n.setWeights(new double[]{0.1, 0.1});
        n.setBias(0.0);
        n.setLearningRate(0.5);

        double[] inputs = {1, 1};
        double desiredOutput = 1.0;

        n.feed(inputs);
        double errorBefore = Math.abs(desiredOutput - n.getContinuousOutput());

        for (int i = 0; i < 100; i++) {
            n.train(inputs, desiredOutput);
        }

        n.feed(inputs);
        double errorAfter = Math.abs(desiredOutput - n.getContinuousOutput());

        assertTrue("l'erreur doit diminuer apres entrainement", errorAfter < errorBefore);
    }

    /*
    *   La methode testTrainOnORGate prouve que le neurone, une fois
    *   entraine sur la table de verite complete de la porte logique
    *   OR, produit bien les sorties binaires attendues pour chacune
    *   des quatre combinaisons d'entrees.
    */
    @Test
    public void testTrainOnORGate() {

        SigmoidNeuron n = new SigmoidNeuron(2);
        n.setWeights(new double[]{0.1, 0.1});
        n.setBias(0.0);
        n.setLearningRate(0.5);

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[] desiredOutputs = {0, 1, 1, 1};

        for (int epoch = 0; epoch < 5000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                n.train(inputs[i], desiredOutputs[i]);
            }
        }

        assertEquals(" 0 OR 0 should return 0", 0, n.feed(inputs[0]));
        assertEquals(" 0 OR 1 should return 1", 1, n.feed(inputs[1]));
        assertEquals(" 1 OR 0 should return 1", 1, n.feed(inputs[2]));
        assertEquals(" 1 OR 1 should return 1", 1, n.feed(inputs[3]));
    }
}