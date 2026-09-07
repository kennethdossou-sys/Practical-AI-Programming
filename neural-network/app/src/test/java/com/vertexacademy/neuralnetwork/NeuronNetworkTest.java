package com.vertexacademy.neuralnetwork;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

public class NeuronNetworkTest {

    @Test
    public void addLayerAddsFirstLayerWithoutPreviousLayer() {
        NeuronNetwork network = new NeuronNetwork();
        NeuronLayer firstLayer = new NeuronLayer(1, 1);

        network.addLayer(firstLayer);

        assertNull(firstLayer.getPreviousLayer());
        assertNull(firstLayer.getNextLayer());
    }

    @Test
    public void addLayerChainsLayersInOrder() {
        NeuronNetwork network = new NeuronNetwork();
        NeuronLayer firstLayer = new NeuronLayer(1, 1);
        NeuronLayer secondLayer = new NeuronLayer(1, 1);

        network.addLayer(firstLayer);
        network.addLayer(secondLayer);

        assertSame(firstLayer, secondLayer.getPreviousLayer());
        assertSame(secondLayer, firstLayer.getNextLayer());
    }

    @Test(expected = IllegalArgumentException.class)
    public void addLayerRejectsNull() {
        new NeuronNetwork().addLayer(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void addLayerRejectsAlreadyConnectedLayer() {
        NeuronNetwork network = new NeuronNetwork();
        NeuronLayer firstLayer = new NeuronLayer(1, 1);
        NeuronLayer secondLayer = new NeuronLayer(1, 1);

        network.addLayer(firstLayer);
        network.addLayer(secondLayer);

        network.addLayer(secondLayer);
    }

    @Test
    public void networkContainsAnOutputLayer() {
        NeuronNetwork network = new NeuronNetwork();
        NeuronLayer hiddenLayer = new NeuronLayer(2, 1);
        NeuronLayer outputLayer = new NeuronLayer(3, 2);

        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        assertTrue(outputLayer.isOutputLayer());
    }

    @Test
    public void getNumberOfOutputsReturnsOutputLayerNeuronCount() {
        NeuronNetwork network = new NeuronNetwork();
        NeuronLayer outputLayer = new NeuronLayer(3, 1);

        network.addLayer(outputLayer);

        assertEquals(3, network.getNumberOfOutputs());
    }

    @Test(expected = IllegalStateException.class)
    public void getNumberOfOutputsRejectsEmptyNetwork() {
        new NeuronNetwork().getNumberOfOutputs();
    }

    @Test
    public void configureCreatesDigitalComparatorNetwork() {
        NeuronNetwork digitalComparatorNetwork = new NeuronNetwork();

        // Deux entrees, quatre neurones caches et trois sorties :
        // [aGb, aEb, aLb].
        digitalComparatorNetwork.configure(1, 4, 2, 3);

        assertEquals(3, digitalComparatorNetwork.getNumberOfOutputs());
    }

    @Test
    public void configuredDigitalComparatorNetworkCanFeedTwoInputs() {
        NeuronNetwork digitalComparatorNetwork = new NeuronNetwork();
        digitalComparatorNetwork.configure(1, 4, 2, 3);

        int[] outputs = digitalComparatorNetwork.feed(new double[]{0, 1});

        // La methode feed actuelle renvoie 3 si la derniere couche est 
        // la couche de sortie. Autrement dit on aurait 4 valeurs (nb de neurones de la couche cachee)
        assertEquals(3, outputs.length);
    }
}