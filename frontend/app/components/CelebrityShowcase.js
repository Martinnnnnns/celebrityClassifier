'use client';

import Image from 'next/image';

const celebrities = [
  {
    id: 'cristiano_ronaldo',
    name: 'Cristiano Ronaldo',
    image: '/images/ronaldo.jpeg',
    alt: 'Cristiano Ronaldo'
  },
  {
    id: 'lionel_messi',
    name: 'Lionel Messi',
    image: '/images/messi.jpeg',
    alt: 'Lionel Messi'
  },
  {
    id: 'steph_curry',
    name: 'Steph Curry',
    image: '/images/curry.jpeg',
    alt: 'Steph Curry'
  },
  {
    id: 'serena_williams',
    name: 'Serena Williams',
    image: '/images/serena.jpeg',
    alt: 'Serena Williams'
  },
  {
    id: 'carlos_alcaraz',
    name: 'Carlos Alcaraz',
    image: '/images/alcaraz.jpeg',
    alt: 'Carlos Alcaraz'
  }
];

export default function CelebrityShowcase() {
  return (
    <div className="sports-personalities-row">
      {celebrities.map((celebrity) => (
        <div key={celebrity.id} className="card-wrapper" data-player={celebrity.id}>
          <div className="card">
            <div className="custom-circle-image">
              <Image
                src={celebrity.image}
                alt={celebrity.alt}
                width={180}
                height={180}
                className="celebrity-image"
                priority
              />
            </div>
            <div className="card-body">
              <h4 className="card-title">{celebrity.name.toUpperCase()}</h4>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}