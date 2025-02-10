function rad2deg(r){
    return r * 180/Math.PI;
}
function deg2rad(d){
    return d * Math.PI/180;
}
function deg2_3D(d){
   let [polar,alpha] = d;
   alpha = deg2rad(alpha);
   polar = deg2rad(polar);
   
   return [Math.sin(polar) * Math.cos(alpha), Math.sin(polar) * Math.sin(alpha), Math.cos(polar)];
}
function threeD_2_deg(d){
    let [x,y,z] = d;
    let alpha = Math.acos(x / Math.sqrt(x * x + y * y)) * (y < 0 ? -1 : 1);
    let polar = Math.acos(z);
    return [rad2deg(polar), rad2deg(alpha)];
}
function l2_norm_sphere(v){
    let [x,y,z] = v;
    return Math.sqrt(x * x + y * y + z * z);
}

class SphericalVis {
    #nodeRadiusLarge = 1.9 + (1.9 * 0.5);
    #nodeRadiusSmall = 1.9;
    #scaleSpeed = 40;

    #colors = ["#4e79a7","#f28e2c","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab"];
    #margin = {top: 15, bottom: 15, left:15, right:15};

    constructor(svgID, nodes, links, center_node) {
        this.svg = d3.select(svgID);
        this.layer1 = this.svg.append('g');
        this.width = this.svg.node().getBoundingClientRect().width;
        this.height = this.svg.node().getBoundingClientRect().height;        

        [this.nodes, this.links, this.idMap] = initGraph(nodes,links);

        //Sphere variables
        this.geopath = null;
        this.lambda = null;
        this.phi = null;
        this.projection = null;
        this.graticule = null;

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
        let projection = this.projection = d3.geoOrthographic()
            .scale(this.height / 2)
            .translate([this.width / 2, this.height / 2]);

        let path = this.geopath = d3.geoPath()
                .projection(projection)
                .pointRadius(this.#nodeRadiusLarge);

        this.svg.append('path')
            .attr('id', "sphere")
            .datum({type: "Sphere"})
            .attr("d", path)
            .attr("stroke", "black")
            .attr("stroke-width", 2)
            .attr("fill", "white");

        var tthis = this;
        function zoomed(e){
            let transform = e.transform;
            let r = {
              x: tthis.lambda(transform.x),
              y: tthis.phi(transform.y)
            };
        
            tthis.projection.rotate([r.x, r.y, 0]);
            tthis.svg.selectAll(".graticules").attr("d", tthis.geopath);
            tthis.svg.selectAll(".links").attr("d", tthis.geopath);
            tthis.svg.selectAll(".sites").attr("d", d => tthis.geopath(d.circle()));
            // tthis.draw();

        }            
        let zoom = this.zoom = d3.zoom().on('zoom', zoomed);
        this.svg.call(zoom);        
        this.svg.on("dblclick.zoom", null);     
        this.svg.on("wheel.zoom", null);

        this.lambda = d3.scaleLinear()
            .domain([this.#margin.left, this.width-this.#margin.right])
            .range([0, 180]);
   
        this.phi = d3.scaleLinear()
            .domain([this.#margin.top, this.height-this.#margin.bottom])
            .range([0,-90]);        

        this.graticule = d3.geoGraticule().step([10,10]);

        this.assign_pos();

    }

    calc_center(){
        let three_d_coord = this.nodePos.features.map( n => {
            let [pol, alph] = n.geometry.coordinates
            return deg2_3D([pol, alph]);
        });

        let sum = three_d_coord.reduce( (acc, cur) => {
            return acc.map((acc_val, ind) => acc_val + cur[ind]);
        });

        let avg = sum.map((val) => val / three_d_coord.length);
        let avg_norm = l2_norm_sphere(avg);

        return avg.map((val) => val / avg_norm);
    }

    assign_pos(){
        let points = {
          type: "FeatureCollection",
          features: this.nodes.map((n, i) => {
            return {"coordinates": [rad2deg(n.spherical.x), rad2deg(n.spherical.y)], 
                    "label": n.id, 
                    "circle": d3.geoCircle()
                                .center([rad2deg(n.spherical.x), rad2deg(n.spherical.y)])
                                .radius(this.#nodeRadiusSmall), 
                    "radius": this.#nodeRadiusSmall
                };
          })
        }
        
        let edges = this.links.map( (e,i) => {
            let src = points.features[this.idMap.get(e.source.id)];
            let tgt = points.features[this.idMap.get(e.target.id)];
            let obj = {
                "type": "LineString", 
                "coordinates": [src.coordinates, tgt.coordinates],
                "source": this.nodes[this.idMap.get(e.source.id)],
                "target": this.nodes[this.idMap.get(e.target.id)]
            }
            return obj;
          } );
       
        this.nodePos = points;
        this.linkRoutes = edges;

        // let center = this.calc_center();
        // let dists = this.nodePos.features.map(d => {
        //     let cart = deg2_3D(d.geometry.coordinates);
        //     let diff = cart.map( (val, ind) => val - center[ind]);
        //     let dist = l2_norm_sphere(diff);
        //     return {"dist": dist, "node": d.geometry.coordinates};
        // });
        
        // let init_center = dists[d3.minIndex(dists, d => d.dist)].node;
        
        // let newCenterPixel = {"x": this.lambda.invert(init_center[0]), "y": this.phi.invert(init_center[1])}
        // this.svg.transition(d3.transition().duration(750))
        //     .call(this.zoom.transform, d3.zoomIdentity.translate(newCenterPixel.x, newCenterPixel.y));


    }

    drawGraticule(){
       this.svg.append('g')
            .selectAll('.graticules')
            .data([this.graticule()])
            .join(
                enter => enter.append("path")
                        .attr("class", "graticules")
                        .attr("d", this.geopath),
                update => update.attr("d", this.geopath)
            );
    }

    draw(){
      
        this.svg.select("#sphere").attr("d", this.geopath);

        // this.drawGraticule();
      
        this.svg
            .selectAll(".links")
            .data(this.linkRoutes, (d,i) => i)
            .join(
                enter => enter.append("path")
                    .attr("class", "links default-link")
                    // .style("stroke-width", 1)
                    .attr("d", this.geopath),
                update => update 
                    .attr("d", this.geopath)
            )
            // .attr("id", (d) => {
            // });
      
        this.svg
            .selectAll('.sites')
            .data(this.nodePos.features, (d,i) => d.label)
            .join(
                enter => enter.append('path')
                    .attr("class", "sites default-node")
                    .attr('d', node => this.geopath(node.circle()))
                    .attr('stroke', "black"),
                    // .attr('pointer-events', 'visibleStroke'),
                update => update    
                    .attr("d", node => this.geopath(node.circle()))
            )

    }

    addWheel(){
        this.svg.on("wheel", e => {
            let s = this.projection.scale() + this.#scaleSpeed * Math.sign(-e.deltaY);
            s = Math.min(2000, Math.max(10, s));
            this.projection.scale(s);        

            this.nodePos.features.forEach(n => {
                n.circle.radius(((this.height / 2) * n.radius) / s);
            })

            this.draw();
            this.appendInteraction("zoom")
        })

        function preventScroll(e){
            e.preventDefault();
            e.stopPropagation();
            return false;
        }        

        let visContainer = document.getElementById("visualization-container")
        visContainer.addEventListener('wheel', preventScroll);
    }

    addHover(id_list){
        if (! id_list)
            id_list = [];
        
        var tthis = this;
        this.svg.selectAll(".sites")
            .on("mouseenter", function(e,pnt) {
                d3.select(this).filter(n => !id_list.includes("node_" + n.label)).attr("class", "sites hover-node"); //function(){} syntax has a different "this" which is the svg element attached.
                

                let d = tthis.nodes[tthis.idMap.get(pnt.label)];
                tthis.svg.selectAll(".sites").filter(n => d.neighbors.has(n.label)).filter(n => !id_list.includes("node_" + n.label))
                    .attr("class", "sites hover-neighbor-node"); //We added an adjacency list data structure in preprocessing to make this efficient. 

                tthis.svg.selectAll(".links").filter(e => e.source.id === d.id || e.target.id === d.id)
                    .attr("class", "links hover-link");

                tthis.appendInteraction("hover");
            })
            .on("mouseleave", (e, d) => {
                this.svg.selectAll(".sites").filter(n => !id_list.includes("node_" + n.label))
                    .attr("class", "sites default-node");
                
                this.svg.selectAll(".links")
                .attr("class", "links default-link");
            });
    }

    makeCenter(x,y, duration=750){
        this.svg.transition(d3.transition().duration(duration))
            .call(this.zoom.transform, d3.zoomIdentity.translate(-x,-y))
    }

    addDblclick(){
        this.svg.on("dblclick", e => {
            let [x,y] = d3.pointer(e);
            let newCenter = this.projection.invert([x,y]);
            let newCenterPixel = {"x": this.lambda.invert(newCenter[0]), "y": this.phi.invert(newCenter[1])}
            this.makeCenter(newCenterPixel.x, newCenterPixel.y);

            this.appendInteraction("dblclick");
           })
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
        let [x,y] = this.nodePos.features[this.center_node].coordinates;
        x = x % 180;
        y = y % 90;
        let newCenterPixel = {"x": this.lambda.invert(x), "y": this.phi.invert(Math.abs(y) > 55 ? -y : y)};
        this.makeCenter(newCenterPixel.x, newCenterPixel.y, 0);        
    }

    interact(id_list) {
        this.svg.on("click", () => {
            this.appendInteraction("pan");
        })        
        this.addHover(id_list);
        this.addWheel();
        this.addDblclick();
        this.addResetButton();

        if(this.center_node){
            this.setToCenterNode();
        }
    }

    resetToDefault(){
        this.projection.scale(this.height / 2);        
        this.svg.call(this.zoom.transform, d3.zoomIdentity);
        this.draw();
        this.appendInteraction("reset");

        // if(this.center_node !== null){
        //     this.setToCenterNode();
        // }                
    }

    highlight_question(id_list) {
        d3.selectAll(".sites").filter(n => id_list.includes("node_" + n.label))
            .attr("class", "sites question-node");

        this.nodePos.features.filter(n => id_list.includes(`node_${n.label}`)).forEach(n => {
            n.circle.radius(this.#nodeRadiusLarge)
            n.radius = this.#nodeRadiusLarge;
        });
        this.draw();
    }

}
