import { describe, it, expect } from '@jest/globals';
import { Person, Newspaper } from './app.js';

describe('Market Research', () => {

  it('should create a Person with all properties set correctly', () => {
    const person = new Person(30, 'male', 'urban');
    expect(person).toEqual({
      age: 30,
      gender: 'male',
      area: 'urban'
    });
  });

  // Person under 25, there should be a fashion Supplement
  it('should match person under 25 with Fashion supplement', () => {
    const person = new Person(22, 'male', 'rural');
    const newspaper = new Newspaper('Fashion');
    expect(person.marketResearcher(newspaper)).toBe(true);
  });

  it('should reject person above 25 with Fashion supplement', () => {
    const person = new Person(26, 'male', 'rural');
    const newspaper = new Newspaper('Fashion');
    expect(person.marketResearcher(newspaper)).toBe(false);
  });

  // For a women under 40, there should be a fashion supplement 
  it('should match woman under 40 with Fashion supplement', () => {
    const person = new Person(35, 'female', 'rural');
    const newspaper = new Newspaper('Fashion');
    expect(person.marketResearcher(newspaper)).toBe(true);
  });

  it('should reject woman over 40 with Fashion supplement', () => {
    const person = new Person(41, 'female', 'rural');
    const newspaper = new Newspaper('Fashion');
    expect(person.marketResearcher(newspaper)).toBe(false);
  });

  // For men, and for women under 35, there should be a sports supplement 
  it('should match women under 35 with sports supplement', () => {
    const person = new Person(30, 'female', 'rural');
    const newspaper = new Newspaper('Sports');
    expect(person.marketResearcher(newspaper)).toBe(true);
  });

  it('should reject women above 35 with sports supplement', () => {
    const person = new Person(36, 'female', 'rural');
    const newspaper = new Newspaper('Sports');
    expect(person.marketResearcher(newspaper)).toBe(false);
  });

  // For people from urban areas, there should be an entertainment supplement
  it('should match urban people with entertainment supplement', () => {
    const person = new Person(40, 'male', 'urban');
    const newspaper = new Newspaper('Entertainment');
    expect(person.marketResearcher(newspaper)).toBe(true);
  });

  it('should reject rural people with entertainment supplement', () => {
    const person = new Person(40, 'male', 'rural');
    const newspaper = new Newspaper('Entertainment');
    expect(person.marketResearcher(newspaper)).toBe(false);
  });

  // For people over 60, there should be a healthcare supplement
  it('should match people over 60 with Healthcare supplement', () => {
    const person = new Person(65, 'female', 'rural');
    const newspaper = new Newspaper('Healthcare');
    expect(person.marketResearcher(newspaper)).toBe(true);
  });

  it('should reject people under 60 with Healthcare supplement', () => {
    const person = new Person(59, 'female', 'rural');
    const newspaper = new Newspaper('Healthcare');
    expect(person.marketResearcher(newspaper)).toBe(false);
  });
});