from mpi4py import MPI
import scipy.sparse as sp
import itertools
import numpy as np
import CircuitClass_SWTPauliBasis as CClass #Import the circuit class defined in CircuitQED to create circuits.
from CircuitClass_SWTPauliBasis import diagonalize
    
    
def coupling_constant(Heff, nco, rd, pinta = False):
    #Input:
    #Heff = effective matrix computed via SW or other projection technique. It is defined over the qubit resonator.
    #cut off in the number of bosonic modes
    #rd= number of decimals to use when rounding a number.
    
    #Output:
    #P= strength of the coupling. Notice that if the Hq is the Hamiltonian of an harmonic oscillator, the strength of the coupling
    #is already given by Hq[0,1].
    #If pinta, it will print the matrix for the bosonic sector.
    σy = np.array([[0, -1j], [1j, 0]])
    ide = np.identity(nco)
    Hr = (np.kron(σy,ide) @ Heff).reshape((2, nco, 2, nco))
    Hq = np.trace(Hr,axis1=0, axis2=2)/2
    P = Hq[0,1]
    if pinta:
        np.set_printoptions(suppress=True)
        print("Hb/Hb[1,0]=",np.abs(P),"*")
        print(np.round(Hq/np.abs(P),3))
    return P

def SW_effective_Hamiltonian(ψb0, E, ψb):
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


class FluxCavity:
    def __init__(self, γ, alpha, ECq, EJq, ECr, EJr, beta=0, flux1=0, pert1=0, pert2=0, nmax=12, nLC=3):
        # s is ratio between Cjunction and Cqubit. The S is ratio between EJjunction and Ejqubit
        ### s gives Cs/C and S gives EJs/EJ.  
        # This ensures alpha has at least two elements
        circ=CClass.Circuit(5, nmax=nmax)
        # Add capacitances and junctions for qubit. 
        circ.add_capacitance(1.0, 0, 1)
        circ.add_capacitance(1.0, 0, 2)
        circ.add_capacitance(alpha+beta, 1, 2) 
        circ.add_junction(1, 1, +1, 0, -1)
        circ.add_junction(1, 2, -1, 0, +1)
        circ.add_junction(alpha*(1+pert1), 2, -1, 1, +1,-flux1)
        # Add capacitances and junctions for junction.
        circ.add_capacitance(ECq/ECr, 3, 4)
        circ.add_junction(EJr/EJq, 4, -1, 3, 1)
        #Coupling.
        if γ:
            circ.add_capacitance(γ, 2, 3)
        #Ground.
        # Once we remove the ground, we have 3 nodes
        # [0,1] for the qubit, [2] for the cavity
        circ = circ.set_ground([1,4])
        self.EJ = EJq
        self.EC = ECq
        self.ωr = np.sqrt(8 * ECr * EJr)
        
        a = sp.diags(np.sqrt(np.arange(1, nLC+1)), 1)
        n0 = np.sqrt(np.sqrt(EJr/(8*ECr)))
        self.LC_n = n0*(a + a.T)/np.sqrt(2)
        self.LC_E = self.ωr * np.arange(0, nLC+1)
        self.LC_basis = np.eye(nLC+1)
        self.LC_Hamiltonian = sp.diags(self.LC_E)
        self.qubit_Hamiltonian = circ.single_Hamiltonian(nodes=[0,1], EC=self.EC, EJ=self.EJ)
        self.qubit_n = circ._op_kron(2, circ.id, 0, circ.n, 1)
        iq = sp.eye((2*nmax+1)**2)
        ir = sp.eye(nLC+1)
        self.Cinv = np.linalg.inv(circ.C)
        Hint = 4 * self.EC * self.Cinv[1,2] * sp.kron(self.qubit_n, self.LC_n)
        self.Hamiltonian = Hint + Hint.T + sp.kron(self.qubit_Hamiltonian, ir) \
                         + sp.kron(iq, self.LC_Hamiltonian)

#We create a subroutine to implement this circuit.
def Circuit(γ, alpha, beta, s, S, nmax, flux1, EC, EJ, pert1=0, pert2=0):
    ### s gives Cs/C and S gives EJs/EJ.  
    # This ensures alpha has at least two elements
    C=CClass.Circuit(5, nmax=nmax)
    # Add capacitances and junctions for qubit. 
    C.add_capacitance(1.0, 0, 1)
    C.add_capacitance(1.0, 0, 2)
    C.add_capacitance(alpha+beta, 1, 2) 
    C.add_junction(1, 1, +1, 0, -1)
    C.add_junction(1, 2, -1, 0, +1)
    C.add_junction(alpha*(1+pert1), 2, -1, 1, +1,-flux1)
    # Add capacitances and junctions for junction.
    C.add_capacitance(s, 3, 4)
    C.add_junction(S, 4, -1, 3, 1)
    #Coupling.
    if γ:
        C.add_capacitance(γ, 2, 3)
    #Ground.
    return C.set_ground([1,4])

# Subroutine to solve and project a circuit with a Josephson junctin and a resonator coupled via a capacitor. The
# energies labeled with r correspond to resonator and the ones with q to qubits. The nco is the cutoff in the
# bosonic modes.
# Subroutine to solve and project a circuit with a Josephson junction and a resonator coupled via a capacitor. The
# energies labeled with r correspond to resonator and the ones with q to qubits. The nco is the cutoff in the
# bosonic modes. alpha is ratio of junctions in the qubit and gamma strength of the coupling in units of C. For
# full Schreiffel wolf set method = "SW" and for only projection onto the low energy set "QB".
def qubit_res(alpha, EJq, ECq, EJr, ECr, γ, beta = 0, nco =8, pinta = False, method = "SW", fs = "False"):
    ## Result for Φ1=Φ2=0.5Φ0.
    #Set parameters.
    flux1=0.5*2*np.pi
    EJ= EJq
    EC= ECq
    if type(γ) is not list:
        γ =[γ]
#     γ=np.arange(0,0.01,0.01)
    nmax=12
    nLC=nco-1
    rd=10
    e=1e-4

    #Matrix to store the values of the constants.  
    E=np.zeros((len(γ),2*nco))
    Eq=np.zeros((len(γ),2*nco))
    P=np.zeros((len(γ)),dtype=complex)
    Ener_r = np.zeros((len(γ),nLC+1))
    Ener_q = np.zeros((len(γ),2))
    Ener_q_full = np.zeros((len(γ),4))

    #Hamiltonians to construct flux operators.
    Hf_plus = FluxCavity(0, alpha, ECq, EJq, ECr, EJr, beta=beta, flux1=flux1+e*2*np.pi, nmax=nmax, nLC=nLC)
    Hf_plus = Hf_plus.qubit_Hamiltonian
    Hf_minus = FluxCavity(0, alpha, ECq, EJq, ECr, EJr, beta=beta, flux1=flux1-e*2*np.pi, nmax=nmax, nLC=nLC)
    Hf_minus = Hf_minus.qubit_Hamiltonian

    # The derivative of the energy w.r.t. the flux is flux operator because it acts on the +/- superposition states of currents
    F = (Hf_plus - Hf_minus)/(2*e*2*np.pi)


    for j,g in enumerate(γ):
        #Circuit.
        C = FluxCavity(g, alpha, ECq, EJq, ECr, EJr, beta=beta, flux1=flux1, nmax=nmax, nLC=nLC)

        #Renormalized qubit basis.
        Ener_q_full[j,:], ψ0_1 = diagonalize(C.qubit_Hamiltonian, full=False, neig=4)
        ψ0_1 = ψ0_1[:,:2]
        Phase=ψ0_1[:,0].T.conj() @ F @ ψ0_1[:,1]/np.abs(ψ0_1[:,0].T.conj() @ F @ ψ0_1[:,1])
        ψ0_new_1=ψ0_1
        ψ0_new_1[:,1]=ψ0_1[:,1]/Phase
    
        #Renormalized junction basis.
        Ener_r[j,:], ψ0_2 = C.LC_E, C.LC_basis
        #     Composed basis.
        ψb0=np.kron(ψ0_new_1, ψ0_2)

        #System energies and eigenstates.
        E[j,:], ψ = diagonalize(C.Hamiltonian, full=False, neig=2*nco)
        
        #Effective Hamiltonianian.
        if method == "SW":
            Hq = SW_effective_Hamiltonian(ψb0, E[j,:], ψ, EC=EC, EJ=EJ)
        else:
            raise Exception()
        Eq[j,:],_ = np.linalg.eigh(Hq)
        P[j]=coupling_constant(Hq, nco, 10, pinta=pinta)
    return Ener_r, Ener_q_full, E, Eq, P
#------------------------------------------------
# Task to perform:
def task(γ, al, EJr, rq, rr, nco, EJq = 1, method = "SW"):
    ECr = EJr/ rr
    ECq = EJq /rq
    filedata = 'alpha'+str(al)+'EJq'+str(round(EJq,6))+'rq'+str(round(rq,3))+'EJr'+str(round(EJr,6))+'rr'+str(round(rr,3))+'nco'+str(nco)+'method_'+str(method)+'.dat'
    print(filedata)
    #filemat = 'EJq'+str(EJq)+'ECq'+str(ECq_p)+'EJr'+str(EJr_p)+'ECr'+str(ECr_p)+'nco'+str(nco)+'method_'+str(method)'.mat'
    E_r, E_q, Ef , Ep, P = qubit_res(al, EJq, ECq, EJr, ECr, γ, beta = 0, nco = nco, method = method)
    Pc = np.abs(P)
    delta_r = np.array([E_r[i][1]-E_r[i][0] for i in range(len(γ))]) 
    delta_q = np.array([E_q[i][1]-E_q[i][0] for i in range(len(γ))]) 
    f=open(filedata,'w')
    #data_w = np.array([γ] + [Pc] + [delta_r_full] + [delta_q_full])
    np.savetxt(filedata, np.c_[γ, Pc, delta_q, delta_r], header='gamma, coupling, gap qubit, gap resonator ') 
    f.close()
    return
    
#------------------------------------------------
#Paralell definitions:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#List of parameters to diagonalize. Each element of this list is a task to perform. The function task should compute
#the problem for this parameters [EJ,EC,alpha,st,nmax,gamma_max] and stroe in separate files.
alpha = [0.65]
EJq = 1   # unit of energy
rq = [10]
#ECq = [EJq/x for x in rq]  # Charging energy qubit large junction
rEj = [20]
EJr = [EJq/x for x in rEj ]# Josephson energy for the single junction
rr = [10]
γ = [0.1*i for i in range(100)]
nco = [10]
method = "SW"


pa = list(itertools.product(alpha, EJr, rq, rr,nco))
print(*pa[0])
nt_rk = int(len(pa)/(size))
#print([i for i in range(nt_rk*rank, min(len(pa),nt_rk*(rank+1)))])
pa_lo = [task(γ,*pa[i]) for i in range(nt_rk*rank, min(len(pa),nt_rk*(rank+1)))]
#x = task(pa_lo)
quit()  
