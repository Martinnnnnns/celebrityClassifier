'use client';

import Image from 'next/image';

const celebrityImages = {
  'cristiano_ronaldo': '/images/ronaldo.jpeg',
  'lionel_messi': '/images/messi.jpeg',
  'steph_curry': '/images/curry.jpeg',
  'serena_williams': '/images/serena.jpeg',
  'carlos_alcaraz': '/images/alcaraz.jpeg'
};

const celebrityNames = {
  'cristiano_ronaldo': 'Cristiano Ronaldo',
  'lionel_messi': 'Lionel Messi',
  'steph_curry': 'Steph Curry',
  'serena_williams': 'Serena Williams',
  'carlos_alcaraz': 'Carlos Alcaraz'
};

export default function ClassificationResults({ result }) {
  if (!result) return null;

  const winnerClass = result.class;
  const classDictionary = result.class_dictionary;
  const probabilities = result.class_probability;

  const probabilityData = Object.entries(classDictionary).map(([name, index]) => ({
    name: celebrityNames[name] || name,
    key: name,
    probability: probabilities[index],
    isWinner: name === winnerClass
  }));

  return (
    <div className="results-container">
      {/* Winner Card */}
      <div className="result-holder">
        <div className="winner-card">
          <div className="custom-circle-image winner">
            <Image
              src={celebrityImages[winnerClass]}
              alt={celebrityNames[winnerClass]}
              width={180}
              height={180}
              className="celebrity-image"
            />
          </div>
          <div className="card-body">
            <h3 className="winner-name">{celebrityNames[winnerClass]?.toUpperCase()}</h3>
            <p className="confidence">
              Confidence: {Math.max(...probabilities).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Probability Table */}
      <div className="probability-table-container">
        <table className="probability-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Probability Score</th>
            </tr>
          </thead>
          <tbody>
            {probabilityData
              .sort((a, b) => b.probability - a.probability) 
              .map((item) => (
                <tr 
                  key={item.key} 
                  className={item.isWinner ? 'winner-row' : ''}
                >
                  <td>{item.name}</td>
                  <td>{item.probability.toFixed(1)}%</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}