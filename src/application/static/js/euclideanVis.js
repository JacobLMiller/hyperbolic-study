function getFloatsFromString(s){
    /*
    Given string s, returns a list of substrings intrepretable as floating point numbers. 
    */
    const pattern = /-?\d+(\.\d+)?/g;
    const matches = s.match(pattern);
    if (matches) {
        return matches.map(s => parseFloat(s));
    }else {console.log("ahhhhhhhhh"); return null;}//Probably not good error handling.
}

class EuclideanVis {
    #nodeRadiusLarge = 15;
    #nodeRadiusSmall = 10;

    #colors = ["#4e79a7","#f28e2c","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab"];
    #margin = {top: 15, bottom: 15, left:15, right:15};

    constructor(svgID, nodes, links, center_node) {
        this.svgID = svgID.substring(1);
        this.svg = d3.select(svgID);
        
        [this.nodes, this.links, this.idMap] = initGraph(nodes,links);

        this.layer1 = this.svg.append("g");
        this.width = this.svg.node().getBoundingClientRect().width;
        this.height = this.svg.node().getBoundingClientRect().height;

        this.origin = {
            "x": this.width / 2,
            "y": this.height / 2
        }

        this.interactions = new Array();
        this.appendInteraction("start");

        this.center_node = center_node;

    }
    
    appendInteraction(e){
        this.interactions.push(
            {"time": Date.now().toString(), 
             "event": e}
        );
    }

    dumpJson(){
        return JSON.stringify(this.interactions);
    }

    process(){
        //TODO: Change aspect ratio to square
        let xextent = d3.extent(this.nodes, d => d.euclidean.x);
        let yextent = d3.extent(this.nodes, d => d.euclidean.y);

        let xscale = d3.scaleLinear().domain(xextent).range([this.#margin.left, this.width-this.#margin.right]);
        let yscale = d3.scaleLinear().domain(yextent).range([this.#margin.top, this.height-this.#margin.bottom]);

        this.nodes.forEach(d => {
            d.x = xscale(d.euclidean.x);
            d.y = yscale(d.euclidean.y);
            d.r = this.#nodeRadiusSmall;
        });
        
    }

    draw(){

        this.layer1.selectAll(".links")
            .data(this.links, d => d.source.id + d.target.id)
            .join(
                enter => enter.append("line")
                    .attr("class", "links default-link")
                    // .attr("stroke", "grey")
                    // .attr("stroke-width", 2)
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y), 
                update => update, 
                exit => exit)
            .attr("id", d => {
                return "link_" + d.source.id + "_" + d.target.id;
            });

        this.layer1.selectAll(".nodes")
            .data(this.nodes, d => d.id)
            .join(
                enter => enter.append("circle")
                    .attr("class", "nodes default-node")
                    .attr("stroke", "black")
                    // .attr("fill", this.#colors[0])
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y)
                    .attr("r", d => d.r),
                update => update, 
                exit => exit 
            )
            .attr("id", d => {
                return "node_" + d.id;
            });
    }

    addZoom(){
        this.layer1.attr("transform", d3.zoomIdentity);
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
	        .on('zoom', e => {
                this.layer1.attr('transform', e.transform);
                this.layer1.selectAll(".nodes")
                    .attr("r", d => d.r / e.transform.k);
            });
        this.svg.call(this.zoom);
        this.svg.on("dblclick.zoom", null);        
        // this.svg.on("wheel.zoom", null);
        document.getElementById("visualization-container").addEventListener('wheel', () => this.appendInteraction("zoom"))
    }

    addHover(id_list){
        if (! id_list){
            id_list = [];
        }
        var tthis = this;
        this.layer1.selectAll(".nodes")
            .on("mouseenter", function(e,d) {
                if (!id_list.includes("node_" + d.id)) {
                    d3.select(this).attr("class", "nodes hover-node"); //function(){} syntax has a different "this" which is the svg element attached.
                }
                
                tthis.layer1.selectAll(".nodes").filter(n => d.neighbors.has(n.id) && !id_list.includes("node_" + n.id))
                    .attr("class", "nodes hover-neighbor-node"); //We added an adjacency list data structure in preprocessing to make this efficient. 

                tthis.layer1.selectAll(".links").filter(e => e.source.id === d.id || e.target.id === d.id)
                    .attr("class", "links hover-link");

                tthis.appendInteraction("hover");
            })
            .on("mouseleave", (e, d) => {
                this.layer1.selectAll(".nodes").filter(n => !id_list.includes("node_" + n.id))
                    .attr("class", "nodes default-node");
                
                this.layer1.selectAll(".links")
                    .attr("class", "links default-link");
            });
    }

    makeCenter(x,y, k=1, duration=750){
        let t = d3.transition().duration(duration);
        this.svg.transition(t).call(
            this.zoom.transform, 
            d3.zoomIdentity.translate(x,y).scale(k)
        );
    }

    addDblClick(){
        this.svg.on("dblclick", e => {
            const transform = this.layer1.node().attributes.transform.value.toString();
            let floats = getFloatsFromString(transform);
            let x0 = floats[0];
            let y0 = floats[1];
            let k0 = floats[2];

            let xmove = this.origin.x + x0;
            let ymove = this.origin.y + y0;

            let [x,y] = d3.pointer(e);
            
            this.makeCenter(xmove - x, ymove-y, k0);

            this.appendInteraction("dblclick");


        });
    }

    addResetButton(){
        var resetButton = document.createElement("button")
        resetButton.classList.add("reset-button")
        document.getElementById("navbarToggler").appendChild(resetButton)
        resetButton.appendChild(document.createTextNode('Reset Visualization'));
        resetButton.onclick = () => {
            this.resetToDefault();
        }            
    }

    setToCenterNode(){
        const transform = this.layer1.node().attributes.transform.value.toString();
        let floats = getFloatsFromString(transform);
        let x0 = floats[0];
        let y0 = floats[1];
        let xmove = this.origin.x + x0;
        let ymove = this.origin.y + y0;
        let c_node = this.nodes[this.center_node];

        this.makeCenter(xmove-c_node.x, ymove-c_node.y, 1, 0);     
    }

    interact(id_list){
        this.layer1.on("click", () => {
            this.appendInteraction("pan");
        })
        this.addZoom();
        this.addHover(id_list);
        this.addDblClick();
        this.addResetButton();

        if (this.center_node !== null){
            this.setToCenterNode();
        }
    }

    highlight_question(id_list) {
        this.layer1.selectAll(".nodes").filter(n => id_list.includes("node_" + n.id))
            .attr("class", "nodes question-node")
            .attr("r", d => d.r = this.#nodeRadiusLarge);

    }

    resetToDefault(){
        this.svg.call(this.zoom.transform, d3.zoomIdentity);
        // if(this.center_node){
        //     this.setToCenterNode();
        // }

        this.appendInteraction("reset");
    }
}
