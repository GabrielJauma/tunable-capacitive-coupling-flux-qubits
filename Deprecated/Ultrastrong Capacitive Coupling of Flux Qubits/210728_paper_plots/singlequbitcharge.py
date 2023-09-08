#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
np.set_printoptions( linewidth=1000 )
import timeit
# %load_ext line_profiler


# In[2]:


# Here we define some functions that might be useful.

def decomposition_in_pauli_4x4(A,rd, Print=True):
    '''Performs Pauli decomposition of a 2x2 matrix.
    
    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.
    
    Output:
    P= coefficients such that A = ΣP[i,j]σ_iσ_j where i,j=0, 1, 2, 3. '''
    
    i  = np.eye(2)  #σ_0     
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [i, σx, σy, σz] #array containing the matrices.
    labels = ['I', 'σx', 'σy', 'σz'] # useful to print the result.
    
    P=np.zeros((4,4),dtype=complex) #array to store our results.
    #Loop to obtain each coefficient.
    for i in range(4):
        for j in range(4):
            label = labels[i] + ' \U00002A02' + labels[j]
            S=np.kron(s[i], s[j]) #S_ij=σ_i /otimes σ_j. 
            P[i,j] = np.round(0.25*(np.dot(S.T.conjugate(), A)).trace(),rd) #P[i,j]=(1/4)tr(S_ij^t*A)
            if P[i,j] != 0.0 and Print==True:
                print(" %s\t*\t %s " %(P[i,j], label))
    
    return P

def mkron(A, *Blist):
    """ Performs the tensor product (kronecker multiplication) of two or more operators.
    
    Input: 
    A= first operator to multiply.
    Blist= list containing the other operators to multiplicate (ordered from left to right).
    
    Output:
    A= result of the tensor product of the operators."""
    for B in Blist:
        A = sp.kron(A, B)
    return A

def eigs_sorted(w, v):
    """Sorts the eigenvalues in ascending order and the corresponding eigenvectors.
    
    Input:
    w=array containing the eigenvalues in random order.
    v=array representing the eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].
    
    Output:
    w=array containing the eigenvalues in ascending order.
    v=array representing the eigenvectors."""
    
    ndx=np.argsort(w) #gives the correct order for the numbers in v, from smallest to biggest.
    return w[ndx], v[:,ndx]

def real_eigenvectors(U):
    '''Fixes the phase of a vector.
    
    Input:
    U=vector 
    
    Output:
    U= vector without phase.'''
    
    l=U.shape[0]
    avgz = np.sum(U[:l//2,:]*np.abs(U[:l//2,:])**2, 0)
    avgz = avgz/np.abs(avgz)
    # U(i,j) = U(i,j) / z(j) for all i,j
    U = U * avgz.conj()
    return U

import scipy.linalg


def diagonalize(H, full=True, neig=4, real=False):
    """ Diagonalizes a Hamiltonian obtaining its energies and eigenfunctions

    Input:
    full=false(default)-> numpy, true-> scipy
    neig= number of eigenvalues and eigenvectors to obtain

    Output:
    E=eigenvalues vector, E[i]
    U=eigenvectors matrix, U[:, i]"""

    if full:
        E, U = scipy.linalg.eigh(H.toarray() if sp.issparse(H) else H)
    elif True:
        E, U = sp.linalg.eigsh(H, neig, which='SA')
    else:
        E, U = sp.linalg.eigs(H, neig, which='SR')
        E = E.real

    E, U = eigs_sorted(E,U) #sorts eigenvalues in ascending order (and corresponding eigenstates).
    if real:
        U = real_eigenvectors(U) #extracts the phase of the eigenvectors.

    return E[:neig], U[:,:neig]
    
    
def decomposition_in_pauli_2x2(A,rd, Print=True):
    '''Performs Pauli decomposition of a 2x2 matrix.
    
    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.
    
    Output:
    P= 4 coefficients such that A = P[0]*I + P[1]*σx + P[2]*σy + P[3]*σz'''
    
    #Pauli matrices.
    I  = np.eye(2)       
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [I, σx, σy, σz] #array containing the matrices.
    
    P=np.zeros((4),dtype=complex) #array to store our results.
    #Loop to obtain each coefficient.
    for i in range(4):
        P[i]=np.round(0.5*np.trace(s[i].T.conjugate() @  A), rd) #P[i]=(1/2)tr(σ_i^t*A)
        
    #Results.
    if Print:
        print ( P[0], 'I +', P[1], 'σx +', P[2], 'σy +', P[3], 'σz' )
    return P


# In[3]:


class Circuit(object):

    def __init__(self, nodes, nmax=10):
        """Initiatiates the circuit, providing a base structure for the circuit and implementing some useful operators.
        
        Input:
        self=circuit 
        nodes= number of nodes in the circuit
        nmax= maximal n to take into account (default=10)
        
        Output: - """
        
        self.C = np.zeros((nodes,nodes)) #matrix of capacitances.
        self.junctions = [] #to store Josephson junctions information.
        self.V = np.zeros(nodes)#Vector containing the electrostatic potentials connected to the nodes
        self.nodes = nodes #number of nodes.
        self.nmax = nmax
        self.n = sp.diags(np.arange(-nmax, nmax+1)) #charge operator.
        self.id = sp.eye(2*nmax+1) #identity
        self.Up = sp.diags([np.ones(2*nmax)],[1]) #-exponential operator.
        self.Do = self.Up.T.conjugate() #+exponential operator.

        
    def Hamiltonian(self, EC=1.0, EJ=1.0, which=None):
        """Generates the Hamiltonian of the circuit: H= (1/2)*qC^(-1)^q-\sum_b E_{J,b}*cos((1/ϕ0)(Φb+φb))
        
        Input:
        self=circuit 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        H= Hamiltonian of the system"""
        
        Cinv = np.linalg.inv(self.C) #inverse of the capacitance matrix.
        
        if which is None:
            which = range(self.nodes) #sets 'which' to contain all the nodes in the system.

        #Go over all nodes combinations implementing the interactions between them.
        H = 4*EC*sum(
            Cinv[i,j] * self._op_kron(self.nodes,
                                      self.n-(self.V[i]/(8*EC))*self.id, i,
                                      self.n-(self.V[j]/(8*EC))*self.id, j)
            for i in which
            for j in which
            if np.abs(Cinv[i,j])
        ) - EJ *sum(
            self._v_to_junction(self.nodes, ji, fluxi, vi)
            for ji, fluxi, vi in self.junctions
            if np.any(vi[which])
            if not np.any(np.delete(vi,which))
        )
        return H
        
        
    def single_Hamiltonian(self, nodes, EC=1.0, EJ=1.0):
        """Generates the Hamiltonian of a single element in the circuit: H= (1/2)*qC^(-1)^q-\sum_b E_{J,b}*cos((1/ϕ0)(Φb+φb))
        
        Input:
        self=circuit 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        nodes= vector containing the position of the nodes of the element
        
        Output:
        H= Hamiltonian of the system"""
        
        Cinv = np.linalg.inv(self.C) #inverse of the capacitance matrix.
            
        #Go over all nodes combinations implementing the interactions between them.
        H_single = 4*EC * sum(
            Cinv[i,j] * self._op_kron(len(nodes),self.n-(self.V[i-np.amin(nodes)]/(8*EC))*self.id, i-np.amin(nodes), self.n-(self.V[j-np.amin(nodes)]/(8*EC))*self.id, j-np.amin(nodes))
            for i in nodes
            for j in nodes
            if np.abs(Cinv[i,j])
        ) - EJ *sum(
            self._v_to_junction(len(nodes),ji, fluxi, vi[nodes])
            for ji, fluxi, vi in self.junctions
            if np.any(vi[nodes])
            if not np.any(np.delete(vi,nodes))
        )
        return H_single
    
    
    def NonInteracting_terms(self, nodes1, *nodeslists, EC=1.0, EJ=1.0):
        """Generates the non interacting term of the Hamiltonian of the circuit.
        
        Input:
        self=circuit 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        nodes1= vector containing the position of the nodes for the first circuit element.
        nodeslists= list of vectors containing the position of the nodes for each remaining element.
        
        Output:
        H= Non interacting Hamiltonian"""
        
        H_nonint=self.Hamiltonian(EC=EC, EJ=EJ, which=nodes1)
    
        for nodes in nodeslists:
            H_nonint+=self.Hamiltonian(EC=EC, EJ=EJ, which=nodes)
            
        return H_nonint
    
    
    def charge_operator(self, pos):
        '''Given a circuit obtains its charge and phase exponential operators.
        
        Input:
        self= circuit
        pos= position of one of the nodes (desconected from everything).
        
        Output:
        Q = charge operator
        '''
        
        #We built the charge operator, which is simple number representation.
        oplist1 = [self.id] * self.nodes
        oplist1[pos] = self.n
        Q = mkron(*oplist1)
        
        return Q
    
    
    def current_operator(self, pos_1, sign_1, pos_2, sign_2, alpha, EJ):
        '''Given a circuit obtains its charge and phase exponential operators.
        
        Input:
        self= circuit
        pos_i= position of the nodes connected to the alpha (reference) junction.
        sign_i= sign of the contribution (if the node is connected to the ground =0)
        
        
        Output:
        I = current operator
        '''
        
        #We built the current operator, I \propto sin(sign_1*φ_1+sign_2*φ_2-0.5φ_0)
        oplist = [self.id] * self.nodes
        if sign_2>0:
            oplist[pos_2] = self.Do
        elif sign_2<0:
            oplist[pos_2] = self.Up
            
        if sign_1>0:
            oplist[pos_1] = self.Do
        elif sign_1<0:
            oplist[pos_1] = self.Up

        op = mkron(*oplist)
        I = alpha*EJ*0.5*(op - op.T.conjugate())/1j
        
        return I
    
    
    def qubit_subspaces(self, n0, EC=1.0, EJ=1.0):
        '''Given a circuit obtains its qubits basis and the second subspace basis.
        
        Input: 
        self=circuit (note that this must be the circuit with the qubits in maximum frustation, flux=0.5)
        nodes1= vector containing the position of the nodes for the first circuit element.
        nodeslists= list of vectors containing the position of the nodes for each remaining element.
        #Note that the nodes must be introduced in an specific order.
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        n0=number of levels to take into account (the two first levels form the qubit subspace and the others form the second subspace).
        
        Output:
        ψb= qubit states.
        Eb= qubit states energies.
        ψs= second subspace states.
        Es= second subspace energies.
        '''
    
        #Obtain the qubit basis (kronecker product of the elements two qubit states).
        H_aux = self.Hamiltonian( EC, EJ)
        E, ψ= spl.eigsh(H_aux,k=n0,which='SA') 
        E, ψ = eigs_sorted(E,ψ) #sorts eigenvalues in ascending order (and corresponding eigenstates).
        ψ = real_eigenvectors(ψ) #extracts the phase of the eigenvectors.
        Eb=E[:2]
        ψb=ψ[:,:2]
        Es=E[2:]
        ψs=ψ[:,2:]
            
        return ψb, Eb, ψs, Es
    
    
    def effective_Hamiltonian(self, ψb, EC=1.0, EJ=1.0):
        """Given a circuit obtains its Hamiltonian in the lower subspace (qubit).
        
        Input: 
        self=circuit 
        ψb= qubit basis 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        H_qubit= Hamiltonian in the qubit basis."""
        
        return ψb.T.conj() @ (self.Hamiltonian(EC,EJ) @ ψb)
    
    def effective_operator(self, O, ψb, EC=1.0, EJ=1.0):
        """Given a circuit obtains its Hamiltonian in the lower subspace (qubit).
        
        Input: 
        self=circuit 
        ψb= qubit basis 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        H_qubit= Hamiltonian in the qubit basis."""
        
        return ψb.T.conj() @ (O @ ψb)
    
    def qubit_Hamiltonian_2order(self, ψb, Eb, ψs, Es, H0, EC=1.0, EJ=1.0):
        '''Given a circuit obtains its Hamiltonian in the qubits basis.
        
        Input: 
        self=circuit 
        ψb= qubit states.
        Eb= qubit states energies.
        ψs= second subspace states.
        Es= second subspace energies.
        H0= Hamiltonian without the interaction. (sustuir quizas por NonInteractings_terms, depende de si vamos a salir o no de flux=0.5, de si vamos a meter la renormalización o no...)
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        H_qubit= Hamiltonian in the qubit basis.'''
        
        l=ψb.shape[1]
        H_qubit = np.zeros((l,l),dtype=complex) #matrix to store our results.
        H=self.Hamiltonian(EC,EJ) #Hamiltonian to convert.
        #Loop to obtain each element of the matrix: <O>.
        for i in range(l):
            for j in range(l):
                H_qubit[i,j] = ψb[:,i].T.conj() @ H @ ψb[:,j] + 1/2*sum(
                    (1/(Eb[i]-Es[k])+1/(Eb[j]-Es[k]))*(ψb[:,i].T.conj() @ (H-H0) @ ψs[:,k]) * (ψs[:,k].T.conj() @ (H-H0) @ ψb[:,j])
                    for k in range(ψs.shape[1])
                ) 
        return H_qubit
    
    
    def SW_effective_Hamiltonian(self, ψb0, E, ψb, EC=1.0, EJ=1.0):
        """Given a circuit obtains its effective Hamiltonian in the lower subspace (qubit) using S-W tranformation.
        
        Input: 
        self=circuit 
        E= full system eigenvectors
        ψb= full system eigenstates
        ψb0= qubit basis 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        Hq= Hamiltonian in the qubit basis."""
        
        Q = ψb0.T.conj()@ψb
        U, s, Vh= np.linalg.svd(Q)
        A=U@Vh
        Hq=A@np.diag(E)@A.T.conj()
        
        return Hq
    
    def SW_effective_operator(self, O, ψb0, ψb, EC=1.0, EJ=1.0):
        """Given a circuit obtains its effective Hamiltonian in the lower subspace (qubit) using S-W tranformation.
        
        Input: 
        self=circuit 
        E= full system eigenvectors
        ψb= full system eigenstates
        ψb0= qubit basis 
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        
        Output:
        Hq= Hamiltonian in the qubit basis."""
        
        Q = ψb0.T.conj()@ψb
        U, s, Vh= np.linalg.svd(Q)
        A=U@Vh
        Oq=A@(ψb.T.conj()@O@ψb)@A.T.conj()
        
        return Oq

        
    def diagonalize(self,EC=1.0,EJ=1.0,which=None, full=True, neig=4, single=False):
        """ Diagonalizes the circuit Hamiltonian obtaining its energies and eigenfunctions
        
        Input:
        self=circuit
        full=false(default)-> numpy, true-> scipy
        neig= number of eigenvalues and eigenvectors to obtain
        EC= capacitive energy (default=1.0)
        EJ= Josephson energy (default=1.0)
        which= vector containing the position of the nodes to take into account (useful when we want to obtain the Hamiltonian of a single element) (default=None-> full Hamiltonian)
        single= false(default)-> the Hamiltonian in obtained in the full system space, true-> obtain the reduced single element Hamiltonian
        
        Output:
        E=eigenvalues vector, E[i]
        U=eigenvectors matrix, U[:, i]"""
        if single==False:
            E, ψ = diagonalize(self.Hamiltonian(EC, EJ, which), full=full, neig=neig)  
        else:
            E, ψ = diagonalize(self.single_Hamiltonian(which, EC, EJ), full=full, neig=neig) 
        
        return E, ψ
    
    
    def _op_kron(self, nodes, op1, pos1, op2, pos2):
        """Gives the multiplication of two operators in the same or different subspaces taking into account the subspaces that don't participe.
        
        Input:
        self=circuit
        op1,op2=operators
        pos1,pos2= operators positions (node correspondence)
        
        Output:
        kronecker product"""
        
        oplist = [self.id] * nodes #set subspace operators to identity.
        
        if pos1 == pos2:
            #if operators in the same subspace perform matrix multiplication between them.
            oplist[pos1] = op1 @ op2
        else:
            #if operators in different subspaces perform kronecker multiplication between them.
            oplist[pos1] = op1
            oplist[pos2] = op2
      
        return mkron(*oplist)
    

    def _v_to_junction(self, nodes, J, flux, v):
        """Given the Josephson parameters implements the operator associated with the Josephson term
        
        Input: 
        EJ=Josephson energy
        flux=external flux
        V= positions and signs in the term of the node fluxes 
        
        Output:
        Josephson operator"""
        
        oplist = [self.id] * nodes
        for i, vi in enumerate(v):
            if vi > 0:
                oplist[i] = self.Up
            elif vi < 0:
                oplist[i] = self.Do
        op = mkron(*oplist) * np.exp(1j *flux)
        return 0.5 * J * (op + op.T.conjugate())


    def add_capacitance(self, C12, node1, node2):
        """Add a capacitive interaction between nodes 1 and 2,
        of the form 1/2 * C12 (dφ1/dt - dφ2/dt)^2"""
        self.C[node1,node2] += -C12
        self.C[node2,node1] += -C12
        self.C[node1,node1] += C12
        self.C[node2,node2] += C12
        return self.C
        
        
    def add_junction(self, j, node1, sign1, node2, sign2, flux=0):
        """Add a Josephson energy between nodes 1 and 2, with
        strength j*EJ, external flux=Φ/Φ0 and given signs:
            j* cos(flux + sign1*phi1 + sign2*phi2)
        """
        v = np.zeros(self.nodes)
        v[node1] = sign1
        v[node2] = sign2
        self.junctions.append((j, flux, v))
        
        
    def add_potential(self, node, cg, Vi):
        '''Connects the node to an electrostatic potential(Vi) 
        through certain gate capacitance(cg). Redefines the charge in 
        this node: Q'=Q-cg*Vi'''
        self.V[node]=cg*Vi


    def set_ground(self, nodes):
        """ Connects selected nodes to the ground (φi=0).
        Input:
        self=circuit
        nodes= vector including the nodes to connect to the ground
        Output: 
        new circuit"""
        
        indices = [i for i in range(self.nodes) if not i in nodes]
        output = Circuit(len(indices),self.nmax)
        output.C = self.C[np.ix_(indices, indices)]
        output.V = self.V[indices]
        output.junctions = [(j, flux, v[indices])
                            for j, flux, v in self.junctions]
        output.nodes = len(indices)
        return output
    


# In[4]:


#Circuit for a single qubit.
def single_qubit(alpha, flux, EJ, EC, nmax, pert=0):
    C=Circuit(3, nmax=nmax)
    # Add capacitances and junctions for qubit. 
    C.add_capacitance(1.0, 0, 1)
    C.add_capacitance(1.0, 0, 2)
    C.add_capacitance(alpha, 1, 2) 
    C.add_junction(1, 1, +1, 0, -1)
    C.add_junction(1, 2, -1, 0, +1)
    C.add_junction(alpha*(1+pert), 2, -1, 1, +1,-flux)
    #Ground.
    return C.set_ground([1])





