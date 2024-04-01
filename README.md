# botderiv2024
Bot indices Sinteticos 2024 red neuronal
#property strict

class NeuralNet {
private:
    int m_inputNeurons;
    int m_outputNeurons;
    int m_hiddenLayer1;
    int m_hiddenLayer2;
    double m_weights[INPUT_NEURONS][HIDDEN_LAYER_1];
    double m_bias[INPUT_NEURONS + HIDDEN_LAYER_1 + HIDDEN_LAYER_2];

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidPrime(double x) {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }

    double randomDouble(double min, double max) {
        return min + (max - min) * MathRand() / 32767.0;
    }

public:
    NeuralNet(int inputNeurons, int hiddenLayer1, int hiddenLayer2, int outputNeurons) {
        m_inputNeurons = inputNeurons;
        m_outputNeurons = outputNeurons;
        m_hiddenLayer1 = hiddenLayer1;
        m_hiddenLayer2 = hiddenLayer2;
    }

    void Init() {
        // Inicializar pesos y valores de sesgo aleatorios
        for (int i = 0; i < m_inputNeurons; i++) {
            for (int j = 0; j < m_hiddenLayer1; j++) {
                m_weights[i][j] = randomDouble(-1.0, 1.0);
            }
            m_bias[i] = randomDouble(-1.0, 1.0);
        }

        for (int i = 0; i < m_hiddenLayer1; i++) {
            for (int j = 0; j < m_hiddenLayer2; j++) {
                m_weights[i][j] = randomDouble(-1.0, 1.0);
            }
            m_bias[i] = randomDouble(-1.0, 1.0);
        }

        for (int i = 0; i < m_hiddenLayer2; i++) {
            for (int j = 0; j < m_outputNeurons; j++) {
                m_weights[i][j] = randomDouble(-1.0, 1.0);
            }
            m_bias[i] = randomDouble(-1.0, 1.0);
        }
    }

    void Train(double &inputs, double &targets, int size, int iterations) {
        // Calcular la salida de la red neuronal para los inputs
        double hiddenLayerOutput[m_hiddenLayer1];
        double outputLayerOutput;

        for (int i = 0; i < m_hiddenLayer1; i++) {
            double sum = 0.0;

            for (int j = 0; j < m_inputNeurons; j++) {
                sum += inputs[j] * m_weights[j][i];
            }

            sum += m_bias[i];
            hiddenLayerOutput[i] = sigmoid(sum);
        }

        double sum = 0.0;

        for (int i = 0; i < m_hiddenLayer2; i++) {
            double hiddenLayerSum = 0.0;

            for (int j = 0; j < m_hiddenLayer1; j++) {
                hiddenLayerSum += hiddenLayerOutput[j] * m_weights[j][i];
            }

            hiddenLayerSum += m_bias[i];
            double hiddenLayerActivation = sigmoid(hiddenLayerSum);

            sum += hiddenLayerActivation * m_weights[i][m_outputNeurons];
        }

        sum += m_bias[m_hiddenLayer2];
        outputLayerOutput = sigmoid(sum);

        // Calcular el error de entrenamiento y actualizar los pesos y valores de sesgo
        double error = targets - outputLayerOutput;
        double errorRate = error * sigmoidPrime(outputLayerOutput);

        for (int i = 0; i < m_hiddenLayer2; i++) {
            double delta = errorRate * hiddenLayerOutput[i];
            m_weights[i][m_outputNeurons] += delta;
            m_bias[i] += delta;
        }

        for (int i = 0; i < m_hiddenLayer1; i++) {
            double delta = 0.0;

            for (int j = 0; j < m_hiddenLayer2; j++) {
                delta += errorRate * m_weights[j][m_outputNeurons];
            }

            delta *= sigmoidPrime(hiddenLayerOutput[i]);
            double hiddenLayerError = delta * inputs[i];

            for (int j = 0; j < m_inputNeurons; j++) {
                m_weights[j][i] += hiddenLayerError;
            }

            m_bias[i] += hiddenLayerError;
        }

        // Actualizar los inputs para el próximo paso de entrenamiento
        for (int i = 0; i < m_inputNeurons; i++) {
            inputs[i] = targets[i];
        }
    }

    void MainLoop() {
        while (!IsStopped()) {
            double lastTickBOOM300 = MarketInfo(SYMBOL_BOOM300, MODE_LAST);
            double lastTickCRASH300 = MarketInfo(SYMBOL_CRASH300, MODE_LAST);

            if (lastTickBOOM300 < 0) {
                pointCount += lastTickBOOM
