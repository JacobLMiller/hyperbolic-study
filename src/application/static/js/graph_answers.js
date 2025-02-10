class GraphComputations {
    constructor(V,E){
        this.V = V;
        this.E = E;
    }
    
    computeT2b(deg){
        /*
        Given node v, how many adjacent nodes have degree n? 
        */
       console.log(this.V);
        let answers = this.V.map(v => {
            return [...v.neighbors].map(u => this.V[u].neighbors.size === deg ? 1 : 0).reduce((partial, el) => partial + el, 0);
        })

        console.log(answers);

    }

}