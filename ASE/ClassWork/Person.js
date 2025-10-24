export class Person {
    constructor(age, gender, area) {
        this.age = age;
        this.gender = gender;
        this.area = area;
    }

    marketResearcher(newspaper) {
        const supplement = newspaper.supplement;

        if (this.age < 25 && supplement === 'Fashion') {
            return true;
        }

        if (this.gender === 'female' && this.age < 40 && supplement === 'Fashion') {
            return true;
        }

        if (this.gender === 'male' && supplement === 'Sports') {
            return true;
        }

        if (this.gender === 'female' && this.age < 35 && supplement === 'Sports') {
            return true;
        }

        if (this.area === 'urban' && supplement === 'Entertainment') {
            return true;
        }

        if (this.age > 60 && supplement === 'Healthcare') {
            return true;
        }

        return false;
    }
}